from numpy import array, float32, roll

Mrange = array([2., 5.5]) # x-value domain of any function of M

#sigma_1
poly_4_prior = array([[-100., 100.], [-500., 300], [100,1300], [-1400, -100], [0, 1500]], dtype = float32)
poly_3_prior = array([[-200., 0], [0., 1000], [-1200, 0], [0, 1500]], dtype = float32)

#sigma_2
linear_prior = array([[0,750], [-100, 500]], dtype = float32)
parabola_prior = array([[-100, 2000], [-2500, 750], [-100, 750]], dtype = float32)

#lambda
exponential_prior = array([[-150, 1], [-10, 50], [-1, 150]], dtype = float32)
inverse_prior = array([[-0.05, 0.075], [-0.10, 0.30]], dtype = float32)

prior_dict = {'poly_4':poly_4_prior, 'poly_3':poly_3_prior,
              'linear':linear_prior, 'parabola':parabola_prior,
              'exponential':exponential_prior, 'inverse': inverse_prior}

