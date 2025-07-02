import numpy as np
import tensorflow as tf
from PIL import Image

import model_optical_system as model_lib


@tf.function
def speckle_mask(phase, target_intensity, threshold):
    '''
    

    Parameters
    ----------
    phase : 2D Array
        Phase values in the target area
    target_intensity : 2D Array
        Intensity vales in the target area
    threshold : float
        If the Intensity in the target area is below the threshold, phase-singularities will be ignored

    Returns
    -------
    singularities : 2D Array
        binary speckle mask

    '''
    padded_phase = tf.pad(phase,((0,1),(0,1)))
    shift_0_1 = padded_phase[:-1,1:]
    shift_1_1 = padded_phase[1:,1:]
    shift_1_0 = padded_phase[1:,:-1]

    # calculate line integral 
    difference_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  
    difference_ta = difference_ta.write(0, shift_0_1 - phase)
    difference_ta = difference_ta.write(1, shift_1_1 - shift_0_1 )
    difference_ta = difference_ta.write(2, shift_1_0 - shift_1_1 )
    difference_ta = difference_ta.write(3, phase - shift_1_0 )
    
    difference_array = difference_ta.stack()
    
    # check if the line integral contains an odd amount of jumps that are greater then pi or less then -pi
    vortex_position_array_neg = tf.math.less(difference_array, -np.pi)
    vortex_position_array_pos = tf.math.greater(difference_array, np.pi)
    # if there is an of odd amount of jumps, a phase singularity occurs
    vortex_position_array = tf.cast(vortex_position_array_pos,tf.float32) - tf.cast(vortex_position_array_neg, tf.float32)
    
    singularities = tf.math.reduce_sum(vortex_position_array,0)
    singularities = tf.math.abs(singularities)

    intensity_mask = tf.math.greater(tf.math.abs(target_intensity),tf.cast(threshold,tf.float32))
    # exclude the phase singularities below the threshold
    singularities = singularities*tf.cast(intensity_mask, tf.float32)
    
    # enlarge the position of the speckle from one pixel to 3x3 pixels
    singularities_pad = tf.pad(singularities,((1,1),(1,1)))
    singularities = singularities_pad[:-2,:-2] + singularities_pad[:-2,1:-1] + singularities_pad[:-2,2:] + singularities_pad[1:-1,:-2]+singularities_pad[1:-1,1:-1]+singularities_pad[1:-1,2:]+singularities_pad[2:,:-2]+singularities_pad[2:,1:-1]+singularities_pad[2:,2:]
    return singularities

@tf.function
def loss_speckle(model, slm_phase, target_intensity):

    
    intensity, far_field, _ = model(slm_phase)

    squared_error = tf.square(target_intensity - intensity[0,0,:,:] )
    mean_squared_error = tf.math.reduce_mean( squared_error )

    phase = tf.math.angle(far_field[0,0,:,:])
    
    speckle_pos = speckle_mask( phase=phase, target_intensity=target_intensity, threshold=tf.constant(0.1*0.2))
    
    speckle_pos = tf.cast(speckle_pos,tf.float32) *tf.math.abs(phase)
    # the speckle loss needs to be balanced in regards to the mean squared error, hence the division by 5000000 (needs some manual fine-tuning)
    speckle_loss = tf.reduce_sum(speckle_pos)/5000000
    

    return speckle_loss + mean_squared_error


def SGD_with_model_speckle(model, initial_phase, image_intensity, loss_list, iterations):
    '''
    

    Parameters
    ----------
    model : 
        tensorflow model of the optical system
    initial_phase : 2D Array
        phase array with values between 0 and 1 (0 = -pi and 1 = pi)
    image_intensity : 2D Array
        Target intensity in the far field
    loss_list : List
        list to store the loss values
    iterations : int
        number of itrations

    Returns
    -------
    computed_phase : 2D Array
        phase-only hologram
    intensity_far_field_target_area : 2D Array 
        Intensity in the target area
    far_field_tilt : 2D Array
        complex field in the far field in the target area with tilt
    far_field_full_no_tilt : 2D Array
        complex field in the far field without the tilt

    '''

    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=.01,
        decay_steps=50,
        decay_rate=0.9,
        staircase=False)


    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)


    initial_phase_expand = np.expand_dims(np.expand_dims(initial_phase,0),0)
    

    slm_phase = tf.Variable(initial_phase_expand, trainable=True, name='phase',constraint=lambda z: tf.math.floormod(z, 1), dtype = tf.float32)
    target_intensity = tf.constant(image_intensity)

    for j in range(iterations):
        with tf.GradientTape() as tp:
            loss_fn = loss_speckle(model, slm_phase, target_intensity)
        gradients = tp.gradient(loss_fn, [slm_phase])
        opt.apply_gradients(zip(gradients, [slm_phase]))
        
        current_loss = loss_fn.numpy()
        loss_list.append(current_loss)
        if (j+1) % 10 == 0 or j==0:
            print(j+1)
            print(current_loss)

    prediction = model(slm_phase)

    far_field_full_no_tilt = np.array(prediction[2])
    
    far_field_tilt = np.array(prediction[1])
    
    intensity_far_field_target_area = np.array(prediction[0])

    computed_phase = np.array(slm_phase[0,0,:,:], dtype=np.float32)
    
    
    
    return computed_phase, intensity_far_field_target_area, far_field_tilt, far_field_full_no_tilt


if __name__ == '__main__':
    
    target_intensity = np.asarray(Image.open('ito_logo.png').convert('L').resize((1820,1820)), dtype=np.float32)/255 * 0.5
    
    initial_phase = np.random.rand(1080,1920)*255
    
    propagation_model = model_lib.build_model(input_size=(1, 1080,1920),crop_position = (100,100),output_size = (1820,1820),far_field_size = 8640)

    # tilt angles in radiants
    tilt_angles = [np.array([0]),np.array([15/180*np.pi])]

    propagation_model.layers[5].set_weights(tilt_angles)
    
    loss_list = []
    
    computed_phase, intensity_target_area, far_field_tilt, full_far_field_no_tilt = SGD_with_model_speckle(propagation_model, initial_phase, target_intensity, loss_list = loss_list, iterations = 500)

    import matplotlib.pyplot as plt
    imgplot = plt.imshow(intensity_target_area[0,0,:,:])
    plt.colorbar(imgplot)
    imgplot = plt.imshow(np.abs(far_field_tilt[0,0,:,:]))