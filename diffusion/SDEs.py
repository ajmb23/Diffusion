import torch 
import numpy as np
 
class VE_zero():

    def __init__(self, sigma_min, sigma_max):
        """
        sigma_min: float, min value 
        sigma_max: float, max value

        defines: 
        1) sigma min value to be called
        using self

        2)geometric "series" as a 
        function of time     
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_t = lambda t: sigma_min * ( sigma_max / sigma_min) ** t

    def pert_mshift(self):
        """
        returns: shifting factor of the mean of the SDE's
        perturbation kernel as a function of time
        Here its just 1, there is no shift
        """
        unity = lambda t: (t+1.)/(t+1.)
        return unity
    
    def pert_std(self):
        """
        returns: standard devitaion of the SDE's
        perturbation kernel as a function of time 
        """
        var_min = (self.sigma_min)**2
        var_t = lambda t: ( self.sigma_t(t) )**2  

        std_diff = lambda t: ( var_t(t) - var_min )**0.5     
        return std_diff

    def drift_coeff(self):
        """
        returns: drift coefficient for specific SDE
        """
        zero = lambda t: 0.*t
        return zero

    def diffusion_coeff(self):
        """
        returns: diffusion coefficient for specific SDE
        """
        diffusion = lambda t: self.sigma_t(t) * np.sqrt( 2. * ( np.log(self.sigma_max) - np.log(self.sigma_min) ) )
        return diffusion 
    

class VE():

    def __init__(self, sigma_min, sigma_max):
        """
        sigma_min: float, min value 
        sigma_max: float, max value

        defines: 
        1) sigma min value to be called
        using self

        2)geometric "series" as a 
        function of time     
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_t = lambda t: sigma_min * ( sigma_max / sigma_min) ** t

    def pert_mshift(self):
        """
        returns: shifting factor of the mean of the SDE's
        perturbation kernel as a function of time
        Here its just 1, there is no shift
        """
        unity = lambda t:  (t+1.)/(t+1.)
        return unity
        
    def pert_std(self):
        """
        returns: standard devitaion of the SDE's
        perturbation kernel as a function of time 
        """   
        return self.sigma_t

    def drift_coeff(self):
        """
        returns: drift coefficient for specific SDE
        """
        zero = lambda t: 0.*t
        return zero
    
    def diffusion_coeff(self):
        """
        returns: diffusion coefficient for specific SDE
        """
        diffusion = lambda t: self.sigma_t(t) * np.sqrt( 2. * ( np.log(self.sigma_max) - np.log(self.sigma_min) ) )
        return diffusion 


class VP():

    def __init__(self, beta_min, beta_max):
        """
        beta_min: float, min value 
        beta_max: float, max value

        defines: 
        1) beta term as a function of t
        beta(t) = beta_min + (beta_max-beta_min)*t
        
        2) analytical integral of beta term  
        for some time t        
        """
        self.beta = lambda t: beta_min + ( beta_max - beta_min) * t
        self.int_beta = lambda t: beta_min * t + 0.5 * ( beta_max - beta_min) * (t**2)

    def pert_mshift(self):
        """
        returns: shifting factor of the mean of the SDE's
        perturbation kernel as a function of time
        """
        shift = lambda t: torch.exp( -0.5 * self.int_beta(t) )
        return shift

    def pert_std(self):
        """
        returns: standard devitaion of the SDE's
        perturbation kernel as a function of time 
        """
        std = lambda t: torch.sqrt( 1. - torch.exp( -self.int_beta(t) ) ) 
        return std

    def drift_coeff(self):
        """
        returns: drift coefficient for specific SDE
        """
        drift_coeff = lambda t: -0.5 * self.beta(t)
        return drift_coeff

    def diffusion_coeff(self):
        """
        returns: diffusion coefficient for specific SDE
        """
        diffusion = lambda t: torch.sqrt( self.beta(t) )
        return diffusion 


class sub_VP():

    def __init__(self, beta_min, beta_max):
        """
        beta_min: float, min value 
        beta_max: float, max value

        defines: 
        1) beta term as a function of t
        beta(t) = beta_min + (beta_max-beta_min)*t
        
        2) analytical integral of beta term  
        for some time t        
        """
        self.beta = lambda t: beta_min + ( beta_max - beta_min) * t
        self.int_beta = lambda t: beta_min * t + 0.5 * ( beta_max - beta_min) * (t**2)

    def pert_mshift(self):
        """
        returns: shifting factor of the mean of the SDE's
        perturbation kernel as a function of time
        """
        shift_coeff = lambda t: torch.exp( -0.5 * self.int_beta(t) )
        return shift_coeff

    def pert_std(self):
        """
        returns: standard devitaion of the SDE's
        perturbation kernel as a function of time 
        """
        std = lambda t: 1. - torch.exp( -self.int_beta(t) )  
        return std

    def drift_coeff(self):
        """
        returns: drift coefficient for specific SDE
        """
        drift_coeff = lambda t: -0.5 * self.beta(t)
        return drift_coeff

    def diffusion_coeff(self):
        """
        returns: diffusion coefficient for specific SDE
        """
        diffusion = lambda t: torch.sqrt( self.beta(t) * ( 1.-torch.exp( -2.*self.int_beta(t) ) ) )
        return diffusion 