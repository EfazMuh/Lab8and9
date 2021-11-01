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
            while map.closest_distance([x,y],theta) == 0:
                x = np.random.uniform(map.bottom_left[0],map.top_right[0])
                y = np.random.uniform(map.bottom_left[1],map.top_right[1])
                theta = np.random.uniform(0, 2*np.pi)
            self._particles.append(Particle(x, y, theta, ln_p))

    def move_by(self, x, y, theta, sigmaT, sigmaD):
        for p in self._particles:
            #sigmaT ~ .02
            dtheta = theta + np.random.normal(0,sigmaT)
            dx = x + np.random.normal(0,sigmaD)
            dy = y + np.random.normal(0,sigmaD)
            while self.map.closest_distance([p.x + dx, p.y + dy],p.theta + dtheta) == 0:
                dtheta = theta + np.random.normal(0,sigmaT)
                dx = x + np.random.normal(0,sigmaD)
                dy = y + np.random.normal(0,sigmaD)
            p.theta = p.theta + dtheta
            p.x = p.x + dx*np.cos(p.theta)
            p.y = p.y + dy*np.sin(p.theta)


    def measure(self, sonar_reading, sigma):
        probs = []
        for p in self._particles:
            probs.append(np.exp(p.ln_p))
        p_logsum = sp.logsumexp(probs)
        weights = []
        for p in self._particles:
            #print(p.x, p.y, p.theta, sigma, sonar_reading)
            p_sens_loc = np.log(st.norm(sonar_reading,sigma).pdf(self.map.closest_distance([p.x,p.y],p.theta)))
            p.ln_p = p_sens_loc + p.ln_p - p_logsum
            weights.append(np.exp(p.ln_p))
        print(weights)
        sumW = sum(weights)
        weights /= sumW

        for p in self._particles:
            p.ln_p = np.log(np.exp(p.ln_p)/sumW)

        self._particles = np.random.choice(self._particles,self.num_particles,True,weights)

    def estimate(self):  
        maxValue = max(self._particles, key=lambda x: x.ln_p)
        return maxValue
