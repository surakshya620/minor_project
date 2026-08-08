#define in1 11    // Left Motor Forward
#define in2 10    // Left Motor Backward
#define in3 6     // Right Motor Forward
#define in4 5     // Right Motor Backward

int command;
const int Speed = 60;          // Constant speed (half of 204)
const int TurnSpeed = 50;       // Speed for smooth turning

void setup() {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  Serial.begin(9600);           // HC-05 Bluetooth baud rate
}

void loop() {

  if (Serial.available()) {

    command = Serial.read();

    switch (command) {

      case 'F':
        forward();
        break;

      case 'B':
        back();
        break;

      case 'L':
        left();
        break;

      case 'R':
        right();
        break;

      case 'G':
        forwardleft();
        break;

      case 'I':
        forwardright();
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
  }
}

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