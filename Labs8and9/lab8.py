import lab8_map
import math
from particle_filter import *

class Run:
    def __init__(self, factory):
        """Constructor.

        Args:
            factory (factory.FactoryCreate)
        """
        self.create = factory.create_create()
        self.time = factory.create_time_helper()
        self.servo = factory.create_servo()
        self.sonar = factory.create_sonar()
        # Add the IP-address of your computer here if you run on the robot
        self.virtual_create = factory.create_virtual_create()
        self.map = lab8_map.Map("lab8_map.json")
        self.particles = ParticleFilter(self.map,10)

    def run(self):
        # This is an example on how to visualize the pose of our estimated position
        # where our estimate is that the robot is at (x,y,z)=(0.5,0.5,0.1) with heading pi
        self.virtual_create.set_pose((0.5, 0.5, 0.1), 0)

        # This is an example on how to show particles
        # the format is x,y,z,theta,x,y,z,theta,...
        #data = [0.5, 0.5, 0.1, math.pi/2, 1.5, 1, 0.1, 0]
        data = []


        # This is an example on how to estimate the distance to a wall for the given
        # map, assuming the robot is at (0, 0) and has heading math.pi
        print(self.map.closest_distance((0.5,0.5), 0))

        # This is an example on how to detect that a button was pressed in V-REP
        while True:

            data = []
            b = self.virtual_create.get_last_button()
            if b == self.virtual_create.Button.MoveForward:
                print("Forward pressed!")
                self.create.drive_direct(100,100)
                self.time.sleep(.5)
                self.create.drive_direct(0,0)
                self.particles.move_by(.05,0,0,.01,.01)
            elif b == self.virtual_create.Button.TurnLeft:
                print("Turn Left pressed!")
            elif b == self.virtual_create.Button.TurnRight:
                print("Turn Right pressed!")
            elif b == self.virtual_create.Button.Sense:
                print("Sense pressed!")
                dist = self.sonar.get_distance()
                if dist is not None:
                    self.particles.measure(dist,.1)
            for p in self.particles._particles:
                data.append(p.x)
                data.append(p.y)
                data.append(0.1)
                data.append(p.theta)

            self.virtual_create.set_point_cloud(data)

            self.time.sleep(0.01)
