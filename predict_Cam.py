import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(subject, body):
    sender_email = "sanilj777@gmail.com"
    receiver_email = "vu1s2223013@pvppcoe.ac.in"
    password = "jpye koio cwsm fcct"

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)


# Load the YOLO model
model = YOLO('./runs/detect/train/weights/last.pt')

# Open the webcam (use '0' for the default webcam)
video_capture = cv2.VideoCapture(0)

# Get the frame width and height
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break

    # Apply YOLO object detection
    results = model(frame)[0]

    # Iterate through the detections and draw bounding boxes with labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[cls]} {conf:.2f}'

        # Draw bounding box and label on the frame if confidence is above 0.7
        if conf > 0.7:
            if model.names[cls] == 'leopard':
                send_email("⚠️ RED ALERT: LEOPARD SPOTTED",
                           """Warning! A leopard has been detected nearby.\nPlease evacuate from the premises immediately and seek shelter indoors.\nThis is a safety alert. Contact local authorities if necessary.\nStay safe and take necessary precautions.""")
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame with detections
    cv2.imshow('YOLO Detection', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
