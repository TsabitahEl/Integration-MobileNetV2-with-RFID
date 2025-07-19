#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import serial
import time
import random
import csv
import os

class RFIDSensorPublisher:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        rospy.init_node('rfid_sensor_publisher', anonymous=True)
        self.sensor_pub = rospy.Publisher('/rfid_sensor_data', String, queue_size=10)

        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"📡 ESP32 UART Serial connected on {port}")
            print(f"Publisher topic: /rfid_sensor_data")
        except Exception as e:
            print(f"⚠ Serial connection error: {e}")
            print(f"⚠ Will continue in simulation mode")

        self.run()

    def is_valid(self, epc, rssi):
        try:
            return len(epc.strip()) > 10 and -100 <= float(rssi) <= 0
        except:
            return False

    def run(self):
        rate = rospy.Rate(10)
        print("🚀 RFID Sensor Publisher started - Listening to ESP32")

        while not rospy.is_shutdown():
            if self.ser:
                try:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode(errors='ignore').strip()
                        if not line:
                            continue
                        print(f"📥 Raw data: {line}")

                        if ',' in line:
                            parts = line.split(',')
                            if len(parts) == 2:
                                epc = parts[0].strip()
                                rssi = parts[1].strip()
                                if self.is_valid(epc, rssi):
                                    msg = f"{epc},{rssi}"
                                    self.sensor_pub.publish(msg)
                                    print(f"📤 Published: {msg}")
                                else:
                                    print(f"⚠ Invalid EPC/RSSI format: {epc}, {rssi}")
                            else:
                                print(f"⚠ Too many parts in data: {line}")
                        else:
                            # Simulate if RSSI missing
                            epc = line.strip()
                            if len(epc) > 10:
                                rssi = random.randint(-80, -60)
                                msg = f"{epc},{rssi}"
                                self.sensor_pub.publish(msg)
                                print(f"📤 [simulated RSSI] Published: {msg}")
                            else:
                                print(f"⚠ Malformed EPC only: {line}")
                except Exception as e:
                    print(f"⚠ Serial read error: {e}")
            rate.sleep()

if __name__ == "__main__":
    try:
        possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
        port_to_use = None
        for port in possible_ports:
            try:
                ser = serial.Serial(port, 115200, timeout=0.1)
                ser.close()
                port_to_use = port
                print(f"✅ Found ESP32 on port: {port}")
                break
            except:
                continue
        if port_to_use:
            publisher = RFIDSensorPublisher(port=port_to_use)
        else:
            print("⚠ No ESP32 detected, using fallback port")
            publisher = RFIDSensorPublisher()
    except rospy.ROSInterruptException:
        pass