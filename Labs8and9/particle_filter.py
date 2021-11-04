import numpy as np
import scipy as sc
import scipy.special as sp
import scipy.stats as st

class Particle:
    def __init__(self, x, y, theta, ln_p):
        self.x = x
        self.y = y
        self.theta = theta
        self.ln_p = ln_p

class ParticleFilter:
    def __init__(self, map, num_particles):
        self.map = map
        self.num_particles = num_particles
        self._particles = []
        for _ in range(num_particles):
            x = np.random.uniform(map.bottom_left[0],map.top_right[0])
            y = np.random.uniform(map.bottom_left[1],map.top_right[1])
            theta = np.random.uniform(0, 2*np.pi)
            ln_p = np.log(1/num_particles)
            while map.closest_distance([x,y],theta) == 0 or map.closest_distance([x,y],theta) is None or map.closest_distance([x,y],theta) < 1.4:
                x = np.random.uniform(map.bottom_left[0],map.top_right[0])
                y = np.random.uniform(map.bottom_left[1],map.top_right[1])
                theta = np.random.uniform(0, 2*np.pi)
            self._particles.append(Particle(x, y, theta, ln_p))

    def move_by(self, x, y, theta, sigmaT, sigmaD):
        for p in self._particles:
            #sigmaT ~ .02
            dtheta = theta + np.random.normal(0,sigmaT)
            d = x + np.random.normal(0,sigmaD)
            #dy = y + np.random.normal(0,sigmaD)
            close = self.map.closest_distance([p.x + d*np.cos(p.theta), p.y + d*np.sin(p.theta)], (p.theta + dtheta) % (2 * np.pi))
            closePre = self.map.closest_distance([p.x, p.y], (p.theta + dtheta) % (2 * np.pi))

            print("close", close , "closePre" , closePre, "x", x, "y" , y)

            if (close is None) or (close == 0) or closePre < d or close < d:
                print("NONE OR CLOSE")
                dtheta = 0
                dx = 0
                dy = 0
                d = 0
                """
                dtheta = theta + np.random.normal(0,sigmaT)
                dx = x + np.random.normal(0,sigmaD)
                dy = y + np.random.normal(0,sigmaD)
                close = self.map.closest_distance([p.x + dx, p.y + dy], (p.theta + dtheta) % (2 * np.pi))
                """
            # Removed the multiplication by sin and cos since we already do that to get x and y.
            # Using sin and cos would need original velocity and would assume we're using those to update particle
            # location when we are using the robot's movement to move the particles
            p.theta = (p.theta + dtheta) % (2 * np.pi)
            p.x = p.x + d*np.cos(p.theta)
            p.y = p.y + d*np.sin(p.theta)

    def measure(self, sonar_reading, sigma):
        probs = []
        for p in self._particles:
            probs.append(p.ln_p)
        p_logsum = sp.logsumexp(probs)
        weights = []
        for p in self._particles:
            #print(p.x, p.y, p.theta, sigma, sonar_reading)
            p_sens_loc = np.log(st.norm.pdf(sonar_reading, loc = self.map.closest_distance([p.x,p.y],p.theta), scale = sigma))
            ln_p_prev = p.ln_p
            # Not 100% sure if it should be addition or subtraction but seems to work
            p.ln_p = p_sens_loc +ln_p_prev + p_logsum
            weights.append(np.exp(p.ln_p))
        print(weights)
        sumW = sum(weights)
        weights /= sumW

        new_particles = np.random.choice(self._particles,self.num_particles,True,weights)

        self._particles = []
        psum = 0
        # Need to create new particles since choice returns references to the same particle rather than creating new ones
        for p in new_particles:
            self._particles.append(Particle(p.x, p.y, p.theta, p.ln_p))
            psum += np.exp(p.ln_p)
        for p in self._particles:
            p.ln_p = np.log(np.exp(p.ln_p)/psum)

    def estimate(self):  
        maxValue = max(self._particles, key=lambda x: x.ln_p)
        return maxValue
