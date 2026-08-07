"""
================================================================================
 BLUETOOTH-ONLY TEST TOOL  (no camera, no YOLO -- isolates the serial link)
================================================================================
Run this BY ITSELF first. If this cannot talk to your Arduino, navigate.py
will never be able to either -- so fix it here first, then go back to
navigate.py.

WHAT IT DOES
------------
1. Lists every serial port Windows/Linux/Mac currently sees, so you can
   confirm the exact port name to use.
2. Opens the port you choose and tries to send characters you type, one at a
   time, and prints anything the Arduino sends back (if your sketch echoes).

HOW TO USE
----------
    pip install pyserial
    python bluetooth_test.py

Then type a single letter (f / b / l / r / s) and press Enter to send it.
Watch your Arduino: it should react immediately. Type "exit" to quit.
================================================================================
"""

import sys
import time

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    print("[ERROR] pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)

BAUD_RATE = 9600  # must match the Serial.begin(...) value in your Arduino sketch


def list_available_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("[WARN] No serial ports detected at all.")
        print("       -> Is the HC-05 actually PAIRED (not just visible) in your")
        print("          OS Bluetooth settings? Pairing must show 'Connected',")
        print("          not just 'Paired'.")
        return []

    print("\nAvailable serial ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}   -  {p.description}")
    return ports


def main():
    ports = list_available_ports()

    if ports:
        choice = input(
            "\nType the port name to use exactly as shown above "
            "(e.g. COM6 or /dev/rfcomm0), or press Enter to type it manually: "
        ).strip()
    else:
        choice = ""

    if not choice:
        choice = input("Enter COM port manually (e.g. COM6): ").strip()

    print(f"\n[INFO] Trying to open {choice} at {BAUD_RATE} baud ...")

    try:
        ser = serial.Serial(choice, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {choice}: {e}")
        print("\nCommon causes:")
        print("  1. Wrong port name. On Windows, HC-05 usually creates TWO ports")
        print("     (an 'Outgoing' and 'Incoming') -- you must use the OUTGOING one.")
        print("  2. The port is already open somewhere else (e.g. Arduino IDE's")
        print("     Serial Monitor, or a previous run of this script still open).")
        print("     Close every other program using that port and try again.")
        print("  3. HC-05 is paired but not actually CONNECTED. Re-pair it, and")
        print("     make sure its LED changes from fast-blinking to slow-blinking")
        print("     or solid once connected.")
        print("  4. On Linux: your user may need permission -> ")
        print("     sudo usermod -a -G dialout $USER   (then log out & back in)")
        print("     or the device may need to be bound first with rfcomm.")
        print("  5. Baud rate mismatch with your Arduino sketch's Serial.begin().")
        sys.exit(1)

    time.sleep(2)  # HC-05 needs a moment after the port opens
    print(f"[INFO] Port opened successfully: {choice}")
    print("[INFO] Type a single character (f/b/l/r/s) and press Enter to send it.")
    print("[INFO] Type 'exit' to quit.\n")

    try:
        while True:
            cmd = input("Send > ").strip()
            if cmd.lower() == "exit":
                break
            if not cmd:
                continue

            try:
                ser.write(cmd.encode())
                print(f"[SENT] '{cmd}'")
            except serial.SerialException as e:
                print(f"[ERROR] Write failed: {e}")
                break

            # give the Arduino a moment, then print anything it sent back
            time.sleep(0.2)
            if ser.in_waiting:
                echo = ser.read(ser.in_waiting)
                print(f"[RECEIVED] {echo}")

    finally:
        ser.close()
        print("[INFO] Port closed.")


if __name__ == "__main__":
    main()