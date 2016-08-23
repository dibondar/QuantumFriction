import numpy as np
from scipy import fftpack
from scipy import linalg
from types import MethodType, FunctionType


class QFriction2Markov:
    """
    Find closest representation of the quantum friction in terms of the rate equation.
    """
    def __init__(self, **kwargs):
        """
        The following parameters are to be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a function)
            K - kinetic energy (as a function)
            f - the function determines the friction
            kappa - the constant for quantum friction
            N -
        """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.f
        except AttributeError:
            raise AttributeError("Quantum friction function f(p) is not specified")

        try:
            self.kappa
        except AttributeError:
            raise AttributeError("kappa is not specified")

        try:
            self.N
        except AttributeError:
            raise AttributeError("N is not specified")

        ##########################################################################################
        #
        # Generating grids
        #
        ##########################################################################################

        # get coordinate step size
        self.dX = 2.*self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.X_range = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        # generate momentum range as it corresponds to FFT frequencies
        self.P_range = fftpack.fftfreq(self.X_gridDIM, self.dX/(2*np.pi))

        ##########################################################################################
        #
        # Construct and diagonalize Hamiltonian
        #
        ##########################################################################################

        # Construct the momentum dependent part
        hamiltonian = fftpack.fft(np.diag(self.K(self.P_range)), axis=1, overwrite_x=True)
        hamiltonian = fftpack.ifft(hamiltonian, axis=0, overwrite_x=True)

        # Add diagonal potential energy
        hamiltonian += np.diag(self.V(self.X_range))

        # Diagalize the Hamiltonian
        self.energies, self.eigenstates = linalg.eigh(hamiltonian)

        # extract real part of the energies
        self.energies = np.real(self.energies)

        # covert to the formal convenient for storage
        self.eigenstates = np.ascontiguousarray(self.eigenstates.T.real)

        # take only first few N
        self.energies = self.energies[:self.N]
        self.eigenstates = self.eigenstates[:self.N]

        # normalize each eigenvector
        for psi in self.eigenstates:
            psi /= linalg.norm(psi) * np.sqrt(self.dX)

        ##########################################################################################
        #
        #   Calculate the action of A_plus on eigenstates
        #
        ##########################################################################################

        X = self.X_range[np.newaxis, :]
        P = self.P_range[np.newaxis, :]

        exp_plus = np.exp(0.5j * self.kappa * X)

        A_plus = fftpack.fft(exp_plus * self.eigenstates, axis=1, overwrite_x=True)
        A_plus *= np.sqrt(self.f(0.5 * self.kappa - P) / self.kappa)
        A_plus = fftpack.ifft(A_plus, axis=1, overwrite_x=True)
        A_plus *= exp_plus

        ##########################################################################################
        #
        #   Calculate gamma_plus[i, j]  = abs(sum_k self.eigenstates[i][k] * A_plus[j][k])**2
        #                               = abs(sum_k self.eigenstates[i][k] * A_plus.T[k][j])**2
        #                               = abs(self.eigenstates.dot(A_plus.T))**2
        #
        ##########################################################################################

        gamma_plus = np.abs(self.eigenstates.dot(A_plus.T))**2

        ##########################################################################################
        #
        #   Calculate the action of A_minus on eigenstates
        #
        ##########################################################################################

        exp_minus = exp_plus.conj()

        A_minus = fftpack.fft(exp_minus * self.eigenstates, axis=1, overwrite_x=True)
        A_minus *= np.sqrt(self.f(0.5 * self.kappa + P) / self.kappa)
        A_minus = fftpack.ifft(A_minus, axis=1, overwrite_x=True)
        A_minus *= exp_minus

        ##########################################################################################
        #
        #   Calculate gamma_minus in the same way as gamma_plus
        #
        ##########################################################################################

        gamma_minus = np.abs(self.eigenstates.dot(A_minus.T))**2

        ##########################################################################################
        #
        #   Calculate the total gamma matrix and corresponding energies differences
        #
        ##########################################################################################

        gamma = gamma_minus + gamma_plus
        gamma *= self.dX**2

        spectrum = self.energies[np.newaxis, :] - self.energies[:, np.newaxis]

        indx = np.triu_indices_from(spectrum, 1)
        self.spectrum_upper = spectrum[indx]
        self.gamma_upper = gamma[indx]

        indx = np.tril_indices_from(spectrum, -1)
        self.spectrum_lower = spectrum[indx]
        self.gamma_lower = gamma[indx]

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt # Plotting facility

    print(QFriction2Markov.__doc__)

    # define the system
    sys = QFriction2Markov(
        X_gridDIM=2*512,
        X_amplitude=10.,

        N=20,

        V=lambda _, x: 0.5 * x**2,
        K=lambda _, p: 0.5 * p**2,

        kappa=0.2,

        f=lambda _, p: np.abs(p),
    )

    plt.subplot(121)
    plt.plot(sys.spectrum_lower, sys.gamma_lower, '.r')

    plt.subplot(122)
    plt.plot(sys.spectrum_upper, sys.gamma_upper, '.b')

    plt.show()