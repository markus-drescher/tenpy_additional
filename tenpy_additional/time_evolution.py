"""Additional classes and functions used for time evolution in the new tenpy
"""
# MD, 18.02.2021

import numpy as np

class params_time_evolution:
    """Class to keep the parameters necessary for different time evolution 
       methods.
       
       Parameters
       ----------
       algorithm -- string specifying algorithm used, 
                    e.g. 'tdvp', 'tebd', 'mpoII'
       params -- dict keeping the parameters for the different algorithms

       Attributes
       ----------
       algorithm
       params
    """

    def __init__(self, algorithm, params):
        self.algorithm = algorithm
        self.params = params
        self.test_sanity()

    def test_sanity(self):
        assert('dt' in self.params)
        assert('t_steps_max' in self.params)

        if self.algorithm == 'mpoII' or self.algorithm == 'mpoI':
            assert('mode' in self.params)
            assert('dt_complex' in self.params)

            mode = self.params['mode']
            assert(mode == 'VAR' or mode == 'SVD' or mode == 'SVD_TRUNC')
            if mode == 'SVD_TRUNC':
                assert('m_temp' in self.params)
                m_temp = self.params['m_temp']
                assert(m_temp == 'optimal' or isinstance(m_temp, int))
            
            if mode == 'VAR':
                self.params['mode_new'] = 'variational'
            else:
                self.params['mode_new'] = mode

