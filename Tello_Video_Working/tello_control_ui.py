from PIL import Image
from PIL import ImageTk
import Tkinter as tki
from Tkinter import Toplevel, Scale
import threading
import datetime
import cv2
import os
import time
import platform
import numpy as np
from simple_pid import PID
import time

class TelloUI:
    """Wrapper class to enable the GUI."""


    def __init__(self, tello, outputpath):
        """
        Initial all the element of the GUI,support by Tkinter

        :param tello: class interacts with the Tello drone.

        Raises:
            RuntimeError: If the Tello rejects the attempt to enter command mode.
        """


        self.tello = tello  # videostream device
        self.outputPath = outputpath  # the path that save pictures created by clicking the takeSnapshot button
        self.frame = None  # frame read from h264decoder and used for pose recognition 
        self.thread = None  # thread of the Tkinter mainloop
        self.stopEvent = None
        # control variables
        self.distance = 0.3  # default distance for :sad'move' cmd
        self.degree = 30  # default degree for 'cw' or 'ccw' cmd

        # if the flag is TRUE,the auto-takeoff thread will stop waiting for the response from tello
        self.quit_waiting_flag = False

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create buttons

        self.btn_tracking = tki.Button(
            self.root, text="Tracking", relief="raised", command=self.Tracking)
        self.btn_tracking.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        # self.btn_tracking = tki.Button(self.root, text="Tracking!",
        #                                command=self.Tracking)
        # self.tracking.pack(side="bottom", fill="both",
        #                        expand="yes", padx=10, pady=5)

        self.btn_snapshot = tki.Button(self.root, text="Snapshot!",
                                       command=self.takeSnapshot)
        self.btn_snapshot.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        # self.btn_pause = tki.Button(self.root, text="Pause", relief="raised", command=self.pauseVideo)
        # self.btn_pause.pack(side="bottom", fill="both",
        #                     expand="yes", padx=10, pady=5)

        self.btn_landing = tki.Button(
            self.root, text="Open Command Panel", relief="raised", command=self.openCmdWindow)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

    def _updateGUIImage(self, image):

        """
        Main operation to initial the object of image,and update the GUI panel 
        """
        image = ImageTk.PhotoImage(image)
        # if the panel none ,we need to initial it
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to tello every 5 second
        """

        while True:
            self.tello.send_command('command')
            time.sleep(5)

    def _setQuitWaitingFlag(self):
        """
        set the variable as TRUE,it will stop computer waiting for response from tello  
        """
        self.quit_waiting_flag = True

    def Tracking(self):
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("TELLO Controller")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # the sending_command will send command to tello every 5 seconds
        self.sending_command_thread = threading.Thread(target=self._sendingCommand)

    def videoLoop(self):
        """
        The mainloop thread of Tkinter
        Raises:
            RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """


        try:
            # start the thread that get GUI image and drwa skeleton
            time.sleep(0.5)
            self.sending_command_thread.start()
            while not self.stopEvent.is_set():
                system = platform.system()

                # read the frame for GUI show
                self.frame = self.tello.read()
                if self.frame is None or self.frame.size == 0:
                    continue

                integral_y = 0
                integral_x = 0
                previous_error_y = 0
                start_time = time.time()
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                low_red = np.array([120, 150, 50], np.uint8)
                high_red = np.array([150, 255, 255], np.uint8)
                mask = cv2.inRange(hsv, low_red, high_red)
                # median = cv2.medianBlur(mask, 15)
                font = cv2.FONT_HERSHEY_COMPLEX
                ret, contorno, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                mensaje2 = "Bateria =" + str(self.tellobateria()) + "%"
                # mensaje3 = "Altura =" + str(self.telloheight()) + "cm"
                # cv2.putText(self.frame, mensaje3, (10, 130), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
                if self.tellobateria() > 50:
                    cv2.putText(self.frame, mensaje2, (10, 100), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(self.frame, mensaje2, (10, 100), font, 1, (255, 0, 0), 1, cv2.LINE_AA)

                SET_POINT_X = 960/2
                SET_POINT_Y = 720/2

                cv2.putText(self.frame, '{},{}'.format(SET_POINT_X, SET_POINT_Y), (SET_POINT_X, SET_POINT_Y),
                            font, 0.75, (255, 255, 255), 1,
                            cv2.LINE_AA)
                for cnt in contorno:
                    area = cv2.contourArea(cnt)
                    approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
                    if area > 2000:
                        # start_time = time.time()
                        M = cv2.moments(cnt)
                        if (M["m00"] == 0): M["m00"] = 1
                        x = int(M["m10"] / M["m00"])
                        y = int(M['m01'] / M['m00'])
                        nuevoContorno = cv2.convexHull(cnt)
                        cv2.circle(self.frame, (x, y), 7, (0, 0, 255), -1)
                        cv2.drawContours(self.frame, [nuevoContorno], -1, (0, 255, 0), 5)
                        # start_time = time.time()

                        if len(approx) == 4:
                            delay_pid = time.time() - start_time
                            # start_time = time.time()
                            # tiempo_transcurrido = time.clock() - tiempo_inicial
                            # elapsed_time = time() - start_time
                            mensaje80 = "tiempo =" + str(delay_pid)
                            cv2.putText(self.frame, mensaje80, (10, 90), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            kpy = 0.0022
                            kiy = 0.0008
                            # kdy = 0.02

                            kpx = 0.0022
                            kix = 0.0008

                            # kpx = 0.0022
                            w = 17
                            medidas = cv2.minAreaRect(nuevoContorno)
                            p = medidas[1][0]
                            f = 1000
                            # distancia = (w * f)/p
                            errorx = x - SET_POINT_X
                            errory = y - SET_POINT_Y
                            # mensaje = "Distancia =" + str(round(distancia, 2)) + " cm"
                            # erroryn = errory * kp
                            derivative_y = (errory - previous_error_y) / delay_pid
                            # integral_y = integral_y + errory * delay_pid
                            # pi = kpy * errory + kiy*integral_y
                            integral_y = integral_y + errory * delay_pid
                            integral_x = integral_x + errorx * delay_pid
                            piy = kpy * errory + kiy * integral_y
                            pix = kpx * errorx + kix * integral_x
                            # pi = kp * errory + ki + errory * delay_pid
                            # pid = kpy * errory + kiy * integral_y + kdy * derivative_y
                            mensaje90 = "value pi =" + str(piy)
                            cv2.putText(self.frame, mensaje90, (10, 130), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            # pid = kp * errory + ki + errory * delay_pid + kd * errory / derivative_y

                            if piy >= 0:
                                self.telloDown(piy)
                                previous_error_y = errory
                            else:
                                errory2n = piy*(-1)
                                self.telloUp(errory2n)
                            if pix >= 0:
                                self.telloMoveRight(pix)
                            else:
                                errorx2n = pix*(-1)
                                self.telloMoveLeft(errorx2n)
                            # cv2.putText(self.frame, mensaje, (10, 70), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            # cv2.putText(self.frame, '{},{}'.format(x, y), (x-60, y+45), font, 0.75, (0, 255, 0), 1,
                            #             cv2.LINE_AA)
                            # if distancia > 100:
                            #     distancia2 = distancia - 100
                            #     self.telloMoveForward(distancia2/100)
                            # elif distancia < 80:
                            #     distancia3 = 100 - distancia
                            #     self.telloMoveBackward(distancia3/100)
                image = Image.fromarray(self.frame)
                if system == "Windows" or system == "Linux":

                    self._updateGUIImage(image)

                else:
                    thread_tmp = threading.Thread(target=self._updateGUIImage, args=(image))
                    thread_tmp.start()
                    time.sleep(0.03)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")


    def openCmdWindow(self):
        """
        open the cmd window and initial all the button and text
        """
        panel = Toplevel(self.root)
        panel.wm_title("Command Panel")

        # create text input entry
        text0 = tki.Label(panel,
                          text='This Controller map keyboard inputs to Tello control commands\n'
                               'Adjust the trackbar to reset distance and degree parameter',
                          font='Helvetica 10 bold'
                          )
        text0.pack(side='top')

        text1 = tki.Label(panel, text=
        'W - Move Tello Up\t\t\tArrow Up - Move Tello Forward\n'
        'S - Move Tello Down\t\t\tArrow Down - Move Tello Backward\n'
        'A - Rotate Tello Counter-Clockwise\tArrow Left - Move Tello Left\n'
        'D - Rotate Tello Clockwise\t\tArrow Right - Move Tello Right',
                          justify="left")
        text1.pack(side="top")

        self.btn_landing = tki.Button(
            panel, text="Land", relief="raised", command=self.telloLanding)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.btn_takeoff = tki.Button(
            panel, text="Takeoff", relief="raised", command=self.telloTakeOff)
        self.btn_takeoff.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        # binding arrow keys to drone control
        self.tmp_f = tki.Frame(panel, width=100, height=2)
        self.tmp_f.bind('<KeyPress-w>', self.on_keypress_w)
        self.tmp_f.bind('<KeyPress-s>', self.on_keypress_s)
        self.tmp_f.bind('<KeyPress-a>', self.on_keypress_a)
        self.tmp_f.bind('<KeyPress-d>', self.on_keypress_d)
        self.tmp_f.bind('<KeyPress-Up>', self.on_keypress_up)
        self.tmp_f.bind('<KeyPress-Down>', self.on_keypress_down)
        self.tmp_f.bind('<KeyPress-Left>', self.on_keypress_left)
        self.tmp_f.bind('<KeyPress-Right>', self.on_keypress_right)
        self.tmp_f.pack(side="bottom")
        self.tmp_f.focus_set()

        self.btn_landing = tki.Button(
            panel, text="Flip", relief="raised", command=self.openFlipWindow)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.distance_bar = Scale(panel, from_=0.02, to=5, tickinterval=0.01, digits=3, label='Distance(m)',
                                  resolution=0.01)
        self.distance_bar.set(0.2)
        self.distance_bar.pack(side="left")

        self.btn_distance = tki.Button(panel, text="Reset Distance", relief="raised",
                                       command=self.updateDistancebar,
                                       )
        self.btn_distance.pack(side="left", fill="both",
                               expand="yes", padx=10, pady=5)

        self.degree_bar = Scale(panel, from_=1, to=360, tickinterval=10, label='Degree')
        self.degree_bar.set(30)
        self.degree_bar.pack(side="right")

        self.btn_distance = tki.Button(panel, text="Reset Degreejustin bieber love yourself", relief="raised", command=self.updateDegreebar)
        self.btn_distance.pack(side="right", fill="both",
                               expand="yes", padx=10, pady=5)

    def openFlipWindow(self):
        """
        open the flip window and initial all the button and text
        """
        panel = Toplevel(self.root)
        panel.wm_title("Gesture Recognition")

        self.btn_flipl = tki.Button(
            panel, text="Flip Left", relief="raised", command=self.telloFlip_l)
        self.btn_flipl.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipr = tki.Button(
            panel, text="Flip Right", relief="raised", command=self.telloFlip_r)
        self.btn_flipr.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipf = tki.Button(
            panel, text="Flip Forward", relief="raised", command=self.telloFlip_f)
        self.btn_flipf.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipb = tki.Button(
            panel, text="Flip Backward", relief="raised", command=self.telloFlip_b)
        self.btn_flipb.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

    def takeSnapshot(self):
        """
        save the current frame of the video as a jpg file and put it into outputpath
        """
        # grab the current timestamp and use it to construct the filename
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imshow("gray", self.frame)
        cv2.imwrite(p, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        print("[INFO] saved {}".format(filename))

    # def Tracking(self):

    def tellobateria(self):
        return self.tello.get_battery()

    def telloheight(self):
        return self.tello.get_height()

    def telloTakeOff(self):
        return self.tello.takeoff()

    def telloLanding(self):
        return self.tello.land()

    def telloFlip_l(self):
        return self.tello.flip('l')

    def telloFlip_r(self):
        return self.tello.flip('r')

    def telloFlip_f(self):
        return self.tello.flip('f')

    def telloFlip_b(self):
        return self.tello.flip('b')

    def telloCW(self, degree):
        return self.tello.rotate_cw(degree)

    def telloCCW(self, degree):
        return self.tello.rotate_ccw(degree)

    def telloMoveForward(self, distance):
        return self.tello.move_forward(distance)

    def telloMoveBackward(self, distance):
        return self.tello.move_backward(distance)

    def telloMoveLeft(self, distance):
        return self.tello.move_left(distance)

    def telloMoveRight(self, distance):
        return self.tello.move_right(distance)

    def telloUp(self, dist):
        return self.tello.move_up(dist)

    def telloDown(self, dist):
        return self.tello.move_down(dist)

    def updateTrackBar(self):
        self.my_tello_hand.setThr(self.hand_thr_bar.get())

    def updateDistancebar(self):
        self.distance = self.distance_bar.get()
        print 'reset distance to %.1f' % self.distance

    def updateDegreebar(self):
        self.degree = self.degree_bar.get()
        print 'reset distance to %d' % self.degree

    def on_keypress_w(self, event):
        print "up %d m" % self.distance
        self.telloUp(self.distance)

    def on_keypress_s(self, event):
        print "down %d m" % self.distance
        self.telloDown(self.distance)

    def on_keypress_a(self, event):
        print "ccw %d degree" % self.degree
        self.tello.rotate_ccw(self.degree)

    def on_keypress_d(self, event):
        print "cw %d m" % self.degree
        self.tello.rotate_cw(self.degree)

    def on_keypress_up(self, event):
        print "forward %d m" % self.distance
        self.telloMoveForward(self.distance)

    def on_keypress_down(self, event):
        print "backward %d m" % self.distance
        self.telloMoveBackward(self.distance)

    def on_keypress_left(self, event):
        print "left %d m" % self.distance
        self.telloMoveLeft(self.distance)

    def on_keypress_right(self, event):
        print "right %d m" % self.distance
        self.telloMoveRight(self.distance)

    def on_keypress_enter(self, event):
        if self.frame is not None:
            self.registerFace()
        self.tmp_f.focus_set()

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of
        
        the quit process to continue
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello
        self.root.quit()
