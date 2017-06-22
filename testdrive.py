#!/usr/bin/python
"""
testdrive.py: version 0.1.0

History:
2017/06/19: Initial version
"""

# import some useful modules
import importlib
import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
from PIL import Image
from io import BytesIO
import ast
import websockets
import asyncio

# set up some global variables
model = None
prev_image_array = None
sse = 0.
acte = 0.
datapoints = 0
startx = -40.62
starty = 108.73
laps = 1
loop = None


# Define a class that will handle our generic steering model interface
class CarModel:
    def __init__(self, model_path):
        # a path is given to the model h5 file
        # with it there is model.py that defines the load/save methods
        # we just need to dynamically import the module.
        (path, modelinstance) = os.path.split(model_path)
        model = importlib.import_module("{}.model".format(path))
        self.model = model.Model(model_path)
        print("loading model...")
        self.model.load()
        self.model.kmodel.summary()

    # Define a function that gets the steering prediction
    def get_steering(self, image_array):
        # get the preprocessed image
        image = self.model.preprocess(image_array)
        # store experience from last action
        return self.model.predict(image[None, :, :, :])


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


# Define a function to detect if the lap is done or if the car is off the track
def sim_state(count, x, y, startx, starty, cte):
    # distance from start
    distance = np.sqrt((x - startx)**2+(y - starty)**2)
    # if over 3 meters from center of track - we are done.
    if abs(cte) > 2:
        return True, False
    # if less than half a meter from the start
    elif count > 100 and distance < 5.0:
        return False, True
    return False, False


# create a PID instance and set up the controller
controller = SimplePIController(0.1, 0.002)


# Define a function to convert to rover-centric coordinates
def car_coords(xpos, ypos, x, y, psi):
    # Calculate pixel positions with reference to the rover position being at
    # the center bottom of the image.
    vx = xpos - x
    vy = ypos - y
    vptsx = vx*np.cos(psi) + vy*np.sin(psi)
    vptsy = -vx*np.sin(psi) + vy*np.cos(psi)
    return vptsx, vptsy

# Define a function that will handle websocket message events
async def message(websocket, path):
    global sse
    global datapoints
    global acte
    global startx
    global starty
    global laps
    global loop

    msg = await websocket.recv()
    if msg[0] == '4' and msg[1] == '2' and msg[4:13] == "telemetry":
        msg = msg[15:-1]
        data = ast.literal_eval(msg)
        datapoints += 1

        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]

        # Get the current waypoints
        xpos = np.array(data["ptsx"])
        ypos = np.array(data["ptsy"])
        x = data["x"]
        y = data["y"]
        psi = data["psi"]

        # convert to vehicle space
        vptsx, vptsy = car_coords(xpos, ypos, x, y, psi)

        # calculate cross track error (cte)
        poly = np.polyfit(np.array(vptsx), np.array(vptsy), 3)
        polynomial = np.poly1d(poly)
        mps = 0.44704
        cte = polynomial([1.34])[0]
        offtrack, lapdone = sim_state(datapoints, x, y, startx, starty, cte)

        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image = np.asarray(image)
        steering_angle = float(model.get_steering(image))

        # get the throttle/break setting to maintain our desired speed.
        throttle = controller.update(float(speed))

        # calculate accumulated CTE
        acte += abs(cte)

        # count the laps
        if lapdone:
            laps -= 1

        # if the vehicle is not done with laps and is not off track
        # send the next steering and throttle commands
        if laps > 0 and not offtrack:
            senddata = '42["steer",{"steering_angle":' + \
                       steering_angle.__str__() + ',' + \
                       '"throttle":' + throttle.__str__() + ',' + \
                       '"mpc_x": [],"mpc_y": [],' + \
                       '"next_x": [],"next_y": []}]'

        # else - send a reset to the simulator
        else:
            senddata = '42["reset",{}]'

        # handle websocket send and stop conditions.
        try:
            await websocket.send(senddata)
            print(datapoints, steering_angle, cte, acte)
            if lapdone:
                if laps == 0:
                    loop.stop()
            if offtrack:
                print("Off track detected!")
                loop.stop()

        except:
            pass

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        await websocket.send('42["manual",{}]')


# our main CLI that was initially copied from the Udacity SDC drive.py
if __name__ == '__main__':

    # set up our argument handling.
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--speed',
        type=int,
        default=30,
        help='Cruse Control Speed setting, default 30Mph'
    )
    parser.add_argument(
        '--laps',
        type=int,
        default=1,
        help='Number of laps to drive around the track, default 1 lap'
    )
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the ' +
             'run will be saved.'
    )
    args = parser.parse_args()

    # create the vehicle model interface and load the model.
    model = CarModel(args.model)

    # set the cruze control
    controller.set_desired(args.speed)

    # get the number of laps to drive autonomously - defaults to 1.
    laps = args.laps

    # recording?
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # start websocket server
    start_server = websockets.serve(message, 'localhost', 4567)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    print("starting test drive")
    try:
        loop.run_forever()
    finally:
        try:
            model.close()
            loop.close()
        except:
            pass
