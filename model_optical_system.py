#Allgemeines
import numpy as np
import tensorflow as tf
# nufft for Tilt
from tfkbnufft import tfkbnufft

import matplotlib.pyplot as plt
from PIL import Image

class TiltLayer(tf.keras.layers.Layer):


    def build(self, input_shape):
        self.angle_1 = self.add_weight(shape=(1, ), initializer="zeros", trainable=True, name = 'tilt_angle_1',dtype = tf.float64) #x
        self.angle_2 = self.add_weight(shape=(1, ), initializer="zeros", trainable=True, name = 'tilt_angle_2',dtype=tf.float64) #y
        self.size = input_shape[-3:]
        self.grid = tf.convert_to_tensor(self.create_sampling_grid(*input_shape[-2:]))
        wavelength = 633*10**-9
        spacing = 3.45*10**-6 
        
        max_k = 1/(2*spacing)
        self.max_w = 1/wavelength/max_k*np.pi
        self.w = tf.math.sqrt(1/wavelength**2 - ((self.grid[:,0]/(np.pi))*max_k)**2 - ((self.grid[:,1]/(np.pi))*max_k)**2) /max_k*np.pi

        self.NufftObj = tfkbnufft.kbnufft.KbNufftModule(im_size = input_shape[-2:], grid_size=input_shape[-2:], grad_traj=True)
        
    
    def create_sampling_grid(self, Ny, Nx):
        y_vec = np.flip((np.repeat(np.arange(Nx),Ny)/Nx*2-1)*np.pi)
        x_vec = np.flip((np.tile(np.arange(Nx),Ny)/Ny*2-1)*np.pi)
        M = np.stack([x_vec, y_vec],1)
        return M 
    
    def resample_grid(self, grid, w, angle_1, angle_2, max_w, shift=True, account_for_homography_shrink=True):
        new_points_0 = tf.math.cos(angle_1)* grid[:,0] + tf.math.sin(angle_1)*tf.math.sin(angle_2) * grid[:,1] + tf.math.sin(angle_1)*tf.math.cos(angle_2)*w
        new_points_1 = tf.math.cos(angle_2)*grid[:,1] - tf.math.sin(angle_2)*w
        if shift == True:
            # shift the sampling grid to omit the phase ramp
            new_points_0 = (new_points_0 - max_w* tf.math.sin(angle_1)*tf.math.cos(angle_2) - tf.math.sin(angle_1)*tf.math.sin(angle_2)*np.pi) 
            new_points_1 = (new_points_1 + max_w* tf.math.sin(angle_2)) 
        if account_for_homography_shrink == True:
            # the homography will shrink the recorded image back along the tilt direction, this is simulated here
            new_points_0 = new_points_0/ tf.math.cos(angle_1)
            new_points_1 = new_points_1 / tf.math.cos(angle_2)

        return  tf.stack([new_points_1, new_points_0],0)[None,...]
    
    def rotate(self, field, angle_1, angle_2):
        new_points = self.resample_grid(self.grid, self.w, angle_1, angle_2, max_w = self.max_w)
        freq_area = tf.signal.ifftshift( tf.signal.ifft2d( tf.signal.ifftshift( field )))
        
        if field.shape[0] != None:
            new_points = tf.repeat(new_points, field.shape[0], axis=0)
            freq_flatt = tf.reshape(freq_area,[field.shape[0],-1])[:, tf.newaxis,...]
        else:
            freq_flatt = tf.reshape(freq_area,[-1])[tf.newaxis, tf.newaxis,...]
        y = tfkbnufft.kbnufft_adjoint(self.NufftObj._extract_nufft_interpob())(freq_flatt, new_points)
        
        return y

    def call(self, inputs):
        return self.rotate(inputs, self.angle_1, self.angle_2)
    
class LinearPhaseLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        return inputs*2*np.pi

class UniformAmplitudeLayer(tf.keras.layers.Layer):
    
    def build(self, input_shape):
        
        self.amplitude = tf.ones((1,input_shape[-2],input_shape[-1]))
        #self.amplitude = tf.ones(input_shape)
        self.amplitude_scale = self.add_weight(shape=(), initializer="ones", trainable=True, name = 'amplitude_scale')
        self.amplitude_scale.assign(0.00025)
    
    
    def call(self, inputs, training = False):
        weighted_amplitude = self.amplitude * self.amplitude_scale
        
        return tf.complex(weighted_amplitude, 0.) * tf.complex(tf.math.cos(inputs), tf.math.sin(inputs))

class PropagateFarFieldLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        inputs_shifted = tf.signal.fftshift(inputs)
        
        shifted_imag = tf.math.imag(inputs_shifted)
        shifted_real = tf.math.real(inputs_shifted)
        transformed = tf.keras.ops.fft2((shifted_real, shifted_imag))
        cmplx_field = tf.complex(*transformed)
        
        #cmplx_field = tf.signal.fft2d(inputs_shifted) #for TF version <= 2.9
        return tf.signal.fftshift(cmplx_field)

class IntensityLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.square(tf.abs(inputs))
    
class SensorSaturationLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.clip_by_value(inputs,0,1)



def build_model(input_size=(1, 1080,1920), crop_position = (0,0), output_size = (1920,1920), far_field_size = 8640, name:str = None):
    '''
    input: Phase values from [0;1]
    output: intensity at far field
    '''

    
    pad_0 = int((far_field_size-input_size[-2])/2)
    pad_1 = int((far_field_size-input_size[-1])/2)
    crop_center = int(far_field_size/2)
    
    
    
    puffer = 400 # puffer pixel for tilt
    
    
    
    # --- normal modell ---
    input_phase_grey_value = tf.keras.Input(input_size)
    
    phase_value = LinearPhaseLayer()(input_phase_grey_value)
    
    field_at_slm = UniformAmplitudeLayer()(phase_value)
    

    padded_field_slm = tf.keras.layers.ZeroPadding2D(padding = (pad_0,pad_1),data_format = 'channels_first')(field_at_slm)
    
    
    
    field_far = PropagateFarFieldLayer()(padded_field_slm)
    
    
    field_far_crop = field_far[...,crop_center+crop_position[0]-puffer:crop_center +crop_position[0]+ output_size[0]+puffer, crop_center+crop_position[1]-puffer: crop_center+ crop_position[1]+output_size[1]+puffer]
    
    field_far_tilt = TiltLayer()(field_far_crop)[..., puffer:-puffer, puffer:-puffer]
    
    intensity_far = IntensityLayer()(field_far_tilt)
    
    intensity_far_clipped = SensorSaturationLayer()(intensity_far)

    model = tf.keras.Model(inputs=input_phase_grey_value, outputs=[intensity_far_clipped, field_far_tilt, field_far],name=name)
    return model


if __name__ == '__main__':



    model_combined = build_model(input_size=(1, 1080,1920),crop_position = (100,100),output_size = (1820,1820),far_field_size = 8640)
    model_combined.summary()
    

    phase = np.asarray( Image.open('hologram_ito_speckle_free_no_tilt.png').convert('L') )/255
    
    # tilt angles in radiants
    tilt_angles = [np.array([30/180*np.pi]),np.array([0])]

    model_combined.layers[5].set_weights(tilt_angles)
    

    intensity_far, far_field_tilt, far_field = model_combined(phase[np.newaxis, np.newaxis,:,:])

    imgplot = plt.imshow(intensity_far[0,0,:,:])