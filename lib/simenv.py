#!/usr/bin/python
"""
simenv.py: version 0.1.0

History:
2017/06/19: Initial version converted to a class
"""

# import some useful modules
import argparse
import base64
from datetime import datetime
import shutil
import numpy as np
from PIL import Image
from io import BytesIO
import ast
import websockets
import asyncio
import os
import time


# Define a class that will handle our PID for cruse control
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement
        # integral error
        self.integral += self.error
        return self.Kp * self.error + self.Ki * self.integral


# Define a class that will handle the I/O communications
# and event loop for the simulator environment.
class Drive:
    # Define a start up function to the simulator interface
    def __init__(self, width=320, height=160, speed=30):
        self.startx = -40.62
        self.starty = 108.73
        self.reset(width=width, height=height, speed=speed)

    # Define a reset function to reset the simulator interface
    def reset(self, width=320, height=160,
              speed=30, training=True, testing=False):

        # the reset will depend on the current mode: training or testing
        if not training:
            if not testing:
                if self.complete_lap:
                    print("testing session: ",
                          self.agent.train_count, "completed successfully")
                else:
                    print("testing session: ",
                          self.agent.train_count, "completed in failure")
            else:
                print("training session: ",
                      self.agent.train_count, "completed with corrections")

        # reset the lap feature calculations
        self.cte = 0.
        self.count = 0
        self.is_lap = False
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.input_size = width*height*3
        self.cruz_control = SimplePIController(0.1, 0.002)
        self.set_speed = speed
        self.x = self.startx
        self.y = self.starty
        self.cruz_control.set_desired(self.set_speed)
        self.predicted_steer = 0.
        self.last_action = 0.
        self.correct_steer = 0.
        self.training = training
        self.testing = testing
        self.tse = 0.0
        self.error_count = 0

        # We only do one lap of training at the beginning
        # of a training a generation (g=n+1)
        if not training:

            # we interleave the training sessions with testing sessions
            if testing:
                print("starting testing session: ", self.agent.train_count)

            # until we get to the 40th session and call it quits
            elif self.agent.train_count == 40:
                print("Training Complete!!!!")
                print("Train count:", self.agent.train_count,
                      "final len(X):", len(self.agent.recall.X))
                print("Total Training time (sec): ", self.agent.training_time,
                      "Average time per lap (sec): ",
                      self.agent.training_time/self.agent.train_count)

            # if we were last doing testing, time to do refinement training
            else:
                print("starting training session: ",
                      self.agent.train_count + 1)

    # Define a function to start the websocket connection to the simulator
    # and start the training event loop
    def start_training(self, trainer, agent):
        # allow the event loop access to the trainer and target model agent
        self.trainer = trainer
        self.agent = agent

        # set up the websocket interface
        self.start_server = websockets.serve(self.message, 'localhost', 4567)

        # start up the event loop
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.start_server)
        print("starting training session: 1")

        # run until we finish the training sessions
        try:
            self.loop.run_forever()
        finally:
            try:
                self.agent.stop()
                self.loop.close()
            except:
                pass

    # Define a function to convert to car-centric coordinates
    def _car_coords(self, xpos, ypos, x, y, psi):
        # Calculate pixel positions with reference to the car position being
        # at the center bottom of the image.
        vx = xpos - x
        vy = ypos - y
        vptsx = vx*np.cos(psi) + vy*np.sin(psi)
        vptsy = -vx*np.sin(psi) + vy*np.cos(psi)
        return vptsx, vptsy

    # Define a function to detect if the lap is done or
    # if the vehicle is off the track
    # returns sim_reset_required, lap_complete booleans
    def _sim_state(self):
        # distance from start
        distance = np.sqrt((self.x - self.startx)**2+(self.y - self.starty)**2)
        # if over 3 meters from center of track - we are done.
        if abs(self.cte) > 2:
            return True, False
        # if less than half a meter from the start
        elif self.count > 100 and distance < 5.0:
            return True, True
        # not done yet.
        return False, False

    # Define a function to handle the websocket event messages
    async def message(self, websocket, path):

        # asynchronous i/o wait until we get a message
        # set up exception handling because sockets sometimes
        # throw internal exceptions that we cannot handle and are harmless
        try:
            msg = await websocket.recv()
        except:
            msg = "exception"
            pass

        # skip exceptions messages
        if msg == "exception":
            pass

        # we are only interested in telemetry messages
        # which has the current:
        #  1. steering angle
        #  2. throttle and brake setting combo,
        #  3. speed (valocity) of the vehicle
        #  4. set of 5 nearest waypoints in global coordinates (ptsx,ptsy)
        #  5. current location of the vehicle in global coordinates (x,y)
        #  6. current yaw heading of the vehicle in radians (psi)
        #     in front of the vehicle
        #  7. front camera image (320x160x3)
        elif msg[0] == '4' and msg[1] == '2' and msg[4:13] == "telemetry":
            start_time = time.time()
            self.count += 1
            msg = msg[15:-1]
            data = ast.literal_eval(msg)
            # The current steering angle of the car
            self.steering_angle = float(data["steering_angle"])
            # The current throttle of the car
            self.throttle = float(data["throttle"])
            # The current speed of the car
            self.speed = float(data["speed"])

            # Get the current waypoints
            xpos = np.array(data["ptsx"])
            ypos = np.array(data["ptsy"])
            self.x = data["x"]
            self.y = data["y"]
            psi = data["psi"]

            # convert to vehicle (local) coordinates
            vptsx, vptsy = self._car_coords(xpos, ypos, self.x, self.y, psi)

            # calculate cross track error (cte)
            poly = np.polyfit(np.array(vptsx), np.array(vptsy), 3)
            polynomial = np.poly1d(poly)
            self.cte = polynomial([0.])[0]
            sim_over, self.complete_lap = self._sim_state()

            # Get the current image from the center camera of the car
            imgString = data["image"]
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_array = np.asarray(image)
            preprocess_time = time.time() - start_time

            # get ground truth from trainer
            self.correct_steer = self.trainer.get_steering(image_array)
            trainer_time = time.time() - start_time - preprocess_time
            self.image = self.agent.preprocess(image_array)

            # get predicted steering action from the target model (trainee)
            try:
                self.predicted_steer = float(self.agent.prediction(self.image))
            except:
                pass

            # reset required (lap complete or off-track)
            if sim_over:
                senddata = '42["reset",{}]'

            # business as usual, start the timestep calculations
            else:
                # calculate mean square error
                self.tse += (self.predicted_steer-self.correct_steer)**2
                self.mse = self.tse/self.count

                # if we are at the beginning of the training session,
                # we just want to feed the target model the ground truth
                # from the trainer model
                if self.training:
                    steering_angle = self.correct_steer
                    self.error_count += 1

                # otherwise, we only want to feed the corrections to the target
                # think of this as hard negative mining.
                else:
                    ase = abs(self.correct_steer - self.predicted_steer)
                    if ase > 0.05 and not self.testing:
                        steering_angle = self.correct_steer
                        self.error_count += 1

                    # if we are in testing mode, then we only take the
                    # predicted steering angle from the target model.
                    else:
                        steering_angle = self.predicted_steer

                # update throttle based on current cruse control settings
                throttle = self.cruz_control.update(self.speed)

                # set up the steering and throttle commands back to simulator
                # for this timestep
                senddata = '42["steer",{"steering_angle":' + \
                           steering_angle.__str__() + ',' + \
                           '"throttle":' + throttle.__str__() + ',' + \
                           '"mpc_x": [],"mpc_y": [],' + \
                           '"next_x": [],"next_y": []}]'

                # calculate timestep timing for debugging/profiling.
                total_time = time.time() - start_time

            try:
                # send the command (reset or steering) to the simulator
                await websocket.send(senddata)

                # set up the next training set
                self.agent.train(
                    self.image, self.predicted_steer, self.cte,
                    self.correct_steer, sim_over, self.training,
                    self.testing, self.x, self.y)

                # if the current simulation session is over...
                if sim_over:
                    # reset the simulation interface to is default
                    self.reset(
                        width=self.width, height=self.height,
                        speed=self.set_speed, training=False,
                        testing=(not self.testing))

                    # if we are at the end of the 40th session,
                    # stop the loop
                    if not self.testing and self.agent.train_count == 40:
                        self.loop.stop()
            except:
                pass

        else:
            # NOTE: DON'T EDIT THIS.
            websocket.send('42["manual",{}]')
