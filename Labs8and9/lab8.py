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
        self.virtual_create = factory.create_virtual_create("192.168.1.118")
        self.map = lab8_map.Map("lab8_map.json")
        self.particles = ParticleFilter(self.map, 500)

    def run(self):
        data = []

        while True:

            data = []
            b = self.virtual_create.get_last_button()
            if b == self.virtual_create.Button.MoveForward:
                print("Forward pressed!")
                self.create.drive_direct(100,100)
                self.time.sleep(.5)
                self.create.drive_direct(0,0)
                self.particles.move_by(.1,0,0,.01,.01)
            elif b == self.virtual_create.Button.TurnLeft:
                print("Turn Left pressed!")
                self.create.drive_direct(100,-100)
                self.time.sleep(1.85)
                self.create.drive_direct(0,0)
                self.particles.move_by(0,0,math.pi/2,.01,.01)
            elif b == self.virtual_create.Button.TurnRight:
                print("Turn Right pressed!")
                self.create.drive_direct(-100,100)
                self.time.sleep(1.85)
                self.create.drive_direct(0,0)
                self.particles.move_by(0,0,-math.pi/2,.01,.01)
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
            est = self.particles.estimate()
            self.virtual_create.set_pose((est.x, est.y, 0.1), est.theta)

            self.time.sleep(0.01)
