#define in1 11    // Left Motor Forward
#define in2 10    // Left Motor Backward
#define in3 6     // Right Motor Forward
#define in4 5     // Right Motor Backward

// Ultrasonic Sensor Pins
#define trigPin 8
#define echoPin 9

int command;
const int Speed = 100;
const int TurnSpeed = 60;

long duration;
int distance;

const int OBSTACLE_DIST = 20; // cm

void setup() {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  Serial.begin(9600);   // HC-05 Bluetooth baud rate
}

void loop() {

  distance = getDistance();
  bool obstacle = (distance > 0 && distance <= OBSTACLE_DIST);

  if (Serial.available()) {

    command = Serial.read();

    switch (command) {

      case 'F':
        if (!obstacle) forward();
        else Stop();          // block only forward motion
        break;

      case 'G':
        if (!obstacle) forwardleft();
        else Stop();
        break;

      case 'I':
        if (!obstacle) forwardright();
        else Stop();
        break;

      case 'B':
        back();                // always allowed, even near an obstacle
        break;

      case 'L':
        left();
        break;

      case 'R':
        right();
        break;

      case 'H':
        backleft();
        break;

      case 'J':
        backright();
        break;

      case 'S':
        Stop();
        break;
    }
  } else if (obstacle) {
    // No new command, but too close — stop as a safety default
    Stop();
  }
}

//================== Distance Function ==================

int getDistance() {

  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);

  digitalWrite(trigPin, LOW);

  // Timeout after 30ms (~5m max range) to avoid blocking forever
  duration = pulseIn(echoPin, HIGH, 30000);

  if (duration == 0) {
    return -1; // no echo received / out of range
  }

  distance = duration * 0.034 / 2;
  return distance;
}

//================== Motor Functions ==================

void forward() {
  analogWrite(in1, Speed);
  analogWrite(in2, 0);
  analogWrite(in3, Speed);
  analogWrite(in4, 0);
}

void back() {
  analogWrite(in1, 0);
  analogWrite(in2, Speed);
  analogWrite(in3, 0);
  analogWrite(in4, Speed);
}

void left() {
  analogWrite(in1, 0);
  analogWrite(in2, Speed);
  analogWrite(in3, Speed);
  analogWrite(in4, 0);
}

void right() {
  analogWrite(in1, Speed);
  analogWrite(in2, 0);
  analogWrite(in3, 0);
  analogWrite(in4, Speed);
}

void forwardleft() {
  analogWrite(in1, TurnSpeed);
  analogWrite(in2, 0);
  analogWrite(in3, Speed);
  analogWrite(in4, 0);
}

void forwardright() {
  analogWrite(in1, Speed);
  analogWrite(in2, 0);
  analogWrite(in3, TurnSpeed);
  analogWrite(in4, 0);
}

void backleft() {
  analogWrite(in1, 0);
  analogWrite(in2, TurnSpeed);
  analogWrite(in3, 0);
  analogWrite(in4, Speed);
}

void backright() {
  analogWrite(in1, 0);
  analogWrite(in2, Speed);
  analogWrite(in3, 0);
  analogWrite(in4, TurnSpeed);
}

void Stop() {
  analogWrite(in1, 0);
  analogWrite(in2, 0);
  analogWrite(in3, 0);
  analogWrite(in4, 0);
}