from GL import GL
from DifferentialDriveSimulatedRobot import *
from DR_3DOFDifferentialDrive import *
from math import exp, sqrt, pi
from Pose3D import *
from Histogram import *

def pdf(mean, sigma, x):
    """Compute the PDF for a normal distribution. A lot faster that scipy.stats.norm(mean, sigma).pdf(x)"""
    return 1 / (sigma * sqrt(2 * pi)) * exp(- (x-mean)**2 / (2 * sigma**2))

class GL_3DOFDifferentialDrive(GL, DR_3DOFDifferentialDrive):
    """
    Grid Reckoning Localization for a 3 DOF Differential Drive Mobile Robot.
    """

    def __init__(self, dx_max, dy_max, range_dx, range_dy, p0, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`GL_4DOFAUV` class. Initializes the Dead reckoning localization algorithm as well as the histogram filter algorithm.

        :param dx_max: maximum x displacement in meters
        :param dy_max: maximum y displacement in meters
        :param range_dx: range of x displacements in meters
        :param range_dy: range of y displacements in meters
        :param p0: initial probability histogram
        :param index: index struture containing plotting information
        :param kSteps: number of time steps to simulate the robot motion
        :param robot: robot object
        :param x0: initial robot pose
        :param args: additional arguments
        """

        self.sigma_d = 1

        super().__init__(p0, index, kSteps, robot, x0, *args)

        self.range_dx = range_dx
        self.range_dy = range_dy

        self.Deltax = 0  # x displacement since last x cell motion
        self.Deltay = 0  # y displacement since last y cell motion
        self.Delta_etak = Pose3D(np.zeros((3, 1)))

        self.cell_size = self.pk_1.cell_size_x  # cell size is the same for x and y

        self.uk_1 = np.zeros((2, 1), dtype=np.float32)

    def GetMeasurements(self):
        """
        Read the measurements from the robot. Returns a vector of range distances to the map features.
        Only those features that are within the :attr:`SimulatedRobot.SimulatedRobot.Distance_max_range` of the sensor are returned.
        The measurements arribe at a frequency defined in the :attr:`SimulatedRobot.SimulatedRobot.Distance_feature_reading_frequency` attribute.

        *** To be implemented by the student ***

        :return: vector of distances to the map features
        """

        return self.robot.ReadRanges()

    def StateTransitionProbability_4_uk(self,uk):
        return self.Pk[uk[0, 0], uk[1, 0]]

    def StateTransitionProbability_4_xk_1_uk(self, etak_1, uk):
        """
        Computes the state transition probability histogram given the previous robot pose :math:`\eta_{k-1}` and the input :math:`u_k`:

        *** To be implemented by the student ***

        .. math::

            p(\eta_k | \eta_{k-1}, u_k)

        :param etak_1: previous robot pose in cells
        :param uk: input displacement in number of cells
        :return: state transition probability :math:`p_k=p(\eta_k | \eta_{k-1}, u_k)`

        """
        target_position = etak_1 + uk

        p_k = Histogram2D(self.p0.num_bins_x, self.p0.num_bins_y, self.p0.x_range, self.p0.y_range)

        for x_bin in p_k.x_range:
            for y_bin in p_k.y_range:
                current_position = np.array([[x_bin, y_bin]]).T
                distance = np.linalg.norm(current_position - target_position)

                p_k.element[x_bin, y_bin] = pdf(0, self.sigma_d, distance)

        return p_k
        

    def StateTransitionProbability(self):
        """
        Computes the complete state transition probability matrix. The matrix is a :math:`n_u \times m_u \times n^2` matrix,
        where :math:`n_u` and :math:`m_u` are the number of possible displacements in the x and y axis, respectively, and
        :math:`n` is the number of cells in the map. For each possible displacement :math:`u_k`, each previous robot pose
        :math:`{x_{k-1}}` and each current robot pose :math:`{x_k}`, the probability :math:`p(x_k|x_{k-1},u_k)` is computed.

        *** To be implemented by the student ***

        :return: state transition probability matrix :math:`P_k=p{x_k|x_{k-1},uk}`
        """

        transition_matrix = np.zeros((3, 3, self.p0.nCells, self.p0.nCells))

        for delta_x in range(3):
            for delta_y in range(3):
                uk = np.array([[delta_x - 1, delta_y - 1]]).T
                print("Computing transition matrix for Uk:", uk)

                # n**2 rows, n**2 columns
                p_uk = np.zeros((self.p0.nCells, self.p0.nCells))

                # each column stores the values of p(etak|uk, etak_1)
                for column in range(self.p0.nCells):
                    index_y = column // self.p0.num_bins_x
                    index_x = column % self.p0.num_bins_x

                    etak_1 = np.array([[self.p0.x_range[index_x], self.p0.y_range[index_y]]]).T
                    
                    transition = self.StateTransitionProbability_4_xk_1_uk(etak_1, uk)
                    p_uk[:, column] = transition.histogram_1d

                transition_matrix[delta_x, delta_y] = p_uk

        self.Pk = transition_matrix
        return self.Pk

    def uk2cell(self, uk):
        """"
        Converts the number of cells the robot has displaced along its DOFs in the world N-Frame to an index that can be
        used to acces the state transition probability matrix.

        *** To be implemented by the student ***

        :param uk: vector containing the number of cells the robot has displaced in all the axis of the world N-Frame
        :returns: index: index that can be used to access the state transition probability matrix
        """
        
        uk = np.round(uk).astype(np.int32)
        return uk + np.ones_like(uk)
        

    def MeasurementProbability(self, zk):
        """
        Computes the measurement probability histogram given the robot pose :math:`\eta_k` and the measurement :math:`z_k`.
        In this case the the measurement is the vector of the distances to the landmarks in the map.

        *** To be implemented by the student ***

        :param zk: :math:`z_k=[r_0~r_1~..r_k]` where :math:`r_i` is the distance to the i-th landmark in the map.
        :return: Measurement probability histogram :math:`p_z=p(z_k | \eta_k)`
        """

        total_p_z = Histogram2D(self.p0.num_bins_x, self.p0.num_bins_y, self.p0.x_range, self.p0.y_range)

        for f, feature_distance in zk:
            p_z = Histogram2D(self.p0.num_bins_x, self.p0.num_bins_y, self.p0.x_range, self.p0.y_range)

            for y_bin, y_centre in zip(p_z.y_range, p_z.y_center_range):
                for x_bin, x_centre in zip(p_z.x_range, p_z.x_center_range):
                    cell_centre_position = np.array([[x_centre, y_centre]]).T
                    true_distance = np.linalg.norm(cell_centre_position - f)

                    probability = pdf(true_distance, self.sigma_d, feature_distance)
                    p_z.element[x_bin, y_bin] = probability

            p_z.histogram_1d /= np.sum(p_z.histogram_1d)
            total_p_z.histogram_1d += p_z.histogram_1d

        total_p_z.histogram_1d /= len(zk)
        return total_p_z

    def GetInput(self, usk):
        """
        Provides an implementation for the virtual method :meth:`GL.GetInput`.
        Gets the number of cells the robot has displaced in the x and y directions in the world N-Frame. To do it, it
        calls several times the parent method :meth:`super().GetInput`, corresponding to the Dead Reckoning Localization
        of the robot, until it has displaced at least one cell in any direction.
        Note that an iteration of the robot simulation :meth:`SimulatedRobot.fs` is normally done in the :method:`GL_4DOFAUV.LocalizationLoop`
        method of the :class:`GL_4DOFAUV.Localization` class, but in this case it is done here to simulate the robot motion
        between the consecutive calls to :meth:`super().GetInput`.

        :param usk: control input of the robot simulation
        :return: uk: vector containing the number of cells the robot has displaced in the x and y directions in the world N-Frame
        """

        def absolute_displacement(delta):
            """ Compute the absolute cell displacement based on a metric delta. """
            return abs(delta) // self.cell_size
        
        def true_displacement(delta):
            """ Given a metric delta, compute the corresponding cell displacement. """
            return absolute_displacement(delta) if delta > 0 else -absolute_displacement(delta)


        while absolute_displacement(self.Deltax) + absolute_displacement(self.Deltay) == 0:
            # Run a simulation step
            self.robot.fs(self.robot.xsk, usk)

            # Get encoder readings
            uk = DR_3DOFDifferentialDrive.GetInput(self)
            
            # Convert encoder readings to raw displacement
            delta_reading = (uk - self.uk_1)
            wheel_distance = delta_reading * 2 * pi * self.wheelRadius / self.robot.pulse_x_wheelTurns
            wheel_velocity = wheel_distance / self.dt
            forward_velocity = np.mean(wheel_velocity)
            angular_velocity = ((wheel_velocity - forward_velocity) / (self.wheelBase / 2))[1][0]

            expanded_uk = np.array([[forward_velocity, 0, angular_velocity]]).T

            # Compute displacement
            self.xk = self.xk_1.oplus(expanded_uk * self.dt)
            self.Delta_etak = self.xk - self.xk_1
            
            # Accumulate displacement
            self.Deltax += self.Delta_etak[0][0]
            self.Deltay += self.Delta_etak[1][0]

            self.uk_1 = uk
            self.xk_1 = self.xk

        displacement_x, displacement_y = true_displacement(self.Deltax), true_displacement(self.Deltay)

        # Partially reset deltas (will bring them to [-cell size, cell size])
        self.Deltax -= displacement_x
        self.Deltay -= displacement_y

        return np.array([[displacement_x, displacement_y]]).T
