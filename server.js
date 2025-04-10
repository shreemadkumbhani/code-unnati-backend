require("dotenv").config(); // Load environment variables

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
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.error("❌ MongoDB Connection Error:", err));

// Schema
const EmployeeSchema = new mongoose.Schema({
  name: { type: String, required: true },
  embedding: { type: [Number], required: true },
});
const Employee = mongoose.model("Employee", EmployeeSchema);

// Monkey patch canvas for face-api.js
faceapi.env.monkeyPatch({ Canvas, Image });

// Base64 to Image
function base64ToImage(imageBase64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
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
  console.log("✅ Models loaded successfully");
}
loadModels();

// Extract face embedding
async function getFaceEmbedding(imageBase64) {
  try {
    const img = await base64ToImage(imageBase64);
    const detections = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detections) {
      return Array.from(detections.descriptor);
    } else {
      return null;
    }
  } catch (error) {
    console.error("Error loading image or detecting face:", error);
    return null;
  }
}

// Register face
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

    await Employee.create({ name, embedding });
    res.status(201).json({ message: `Face registered for ${name}` });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ message: "Error registering face" });
  }
});

// Recognize face
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

    const employees = await Employee.find({});
    let recognized = false;

    for (let emp of employees) {
      if (embedding.length !== emp.embedding.length) continue;

      const distance = faceapi.euclideanDistance(embedding, emp.embedding);
      if (distance < 0.6) {
        recognized = true;
        return res.json({ message: `Recognized: ${emp.name}` });
      }
    }

    if (!recognized) {
      return res.status(404).json({ message: "No user found" });
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ message: "Error recognizing face" });
  }
});

// Start server
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
