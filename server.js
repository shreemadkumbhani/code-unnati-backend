require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bodyParser = require("body-parser");
const faceapi = require("face-api.js");
const { Canvas, Image } = require("canvas");

const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: "10mb" }));

// MongoDB Connection
mongoose
  .connect(
    "mongodb+srv://ShreemadKumbhani:shreemad1451@cluster52320.3mu7r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster52320/attendanceDB"
  )
  .then(() => console.log("MongoDB Connected"))
  .catch((err) => console.error("MongoDB Connection Error:", err));

// Schema
const EmployeeSchema = new mongoose.Schema({
  name: { type: String, required: true },
  embedding: { type: [Number], required: true }, // Store face embeddings
});
const Employee = mongoose.model("Employee", EmployeeSchema);

// Set up face-api.js to use the canvas package
faceapi.env.monkeyPatch({ Canvas, Image });

// Function to convert base64 to Image and wait for it to load
function base64ToImage(imageBase64) {
  return new Promise((resolve, reject) => {
    const img = new Image();

    // Ensure the base64 string has the proper prefix
    const base64Prefix = "data:image/png;base64,";
    const fullBase64 = imageBase64.startsWith(base64Prefix)
      ? imageBase64
      : `${base64Prefix}${imageBase64}`;

    img.onload = () => resolve(img);
    img.onerror = (err) => {
      console.error("Error loading image:", err);
      reject(new Error("Unsupported image type or corrupted data"));
    };

    img.src = fullBase64;
  });
}

// Load FaceAPI models
async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  console.log("Models loaded successfully");
}
loadModels();

// Extract Face Embedding
async function getFaceEmbedding(imageBase64) {
  try {
    console.log("Received image:", imageBase64); // Log the base64 image
    const img = await base64ToImage(imageBase64);
    console.log("Image loaded successfully"); // Log image loading success

    const detections = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (detections) {
      console.log("Face detected"); // Log face detection success
      return Array.from(detections.descriptor);
    } else {
      console.log("No face detected in the image"); // Log face detection failure
      return null;
    }
  } catch (error) {
    console.error("Error loading image or detecting face:", error); // Log errors
    return null;
  }
}

// Register Face
app.post("/register", async (req, res) => {
  const { name, image } = req.body;

  if (!name || !image) {
    return res.status(400).json({ message: "Name and image are required" });
  }

  try {
    const embedding = await getFaceEmbedding(image);
    if (!embedding) {
      return res.status(400).json({ message: "No face detected in the image" });
    }

    const newEmployee = await Employee.create({ name, embedding });
    res.status(201).json({ message: `Face registered for ${name}` });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ message: "Error registering face" });
  }
});

// Recognize Face
app.post("/recognize", async (req, res) => {
  const { image } = req.body;

  if (!image) {
    return res.status(400).json({ message: "Image is required" });
  }

  try {
    const embedding = await getFaceEmbedding(image);
    if (!embedding) {
      return res.status(400).json({ message: "No face detected in the image" });
    }

    console.log("Extracted embedding length:", embedding.length); // Log embedding length

    const employees = await Employee.find({});
    let recognized = false;

    for (let emp of employees) {
      console.log("Stored embedding length:", emp.embedding.length); // Log stored embedding length

      if (embedding.length !== emp.embedding.length) {
        console.error(
          "Embedding length mismatch:",
          embedding.length,
          emp.embedding.length
        );
        continue; // Skip this employee
      }

      const distance = faceapi.euclideanDistance(embedding, emp.embedding);
      if (distance < 0.6) {
        // Adjust threshold as needed
        recognized = true;
        return res.json({ message: `Recognized: ${emp.name}` });
      }
    }

    if (!recognized) {
      // If a face is detected but no match is found in the database
      return res.status(404).json({ message: "No user found" });
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ message: "Error recognizing face" });
  }
});

// Start Server
const PORT = 5001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
