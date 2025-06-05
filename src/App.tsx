import React, { useState, useEffect, useRef } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs'; // Registers the CPU backend.
import './App.css';

// Define a type for the detected objects for better type safety
interface DetectedObject extends cocoSsd.DetectedObject {
  // You can extend this if cocoSsd.DetectedObject is missing something or for your own custom props
}

function App(): JSX.Element {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<DetectedObject[]>([]);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(true);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null); 
  const fileInputRef = useRef<HTMLInputElement | null>(null); 
  const MIN_CONFIDENCE = 0.1; // Minimum confidence threshold for displaying detections
  // 1. Load the COCO-SSD model
  useEffect(() => {
    async function loadModel(): Promise<void> {
      try {
        console.log("Loading COCO-SSD model...");
        const loadedModel: cocoSsd.ObjectDetection = await cocoSsd.load();
        setModel(loadedModel);
        setIsLoadingModel(false);
        console.log("Model loaded successfully.");
      } catch (err) {
        console.error("Failed to load model:", err);
        setError("Failed to load AI model. Please try refreshing.");
        setIsLoadingModel(false);
      }
    }
    loadModel();
  }, []);

  // 2. Handle image upload
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0]; // Optional chaining for safety
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        setImageSrc(e.target?.result as string); // Assert result is string
        setDetections([]); // Clear previous detections
        setError(null); // Clear previous errors
      };
      reader.readAsDataURL(file);
    }
  };

  // 3. Perform object detection
  const detectObjects = async (): Promise<void> => {
    if (!model || !imageRef.current) {
      setError("Model or image not ready for detection.");
      return;
    }

    // Ensure image is fully loaded before detection
    if (!imageRef.current.complete || imageRef.current.naturalHeight === 0) {
      console.log("Image not fully loaded yet, waiting for onLoad...");
      // Detection will be triggered by img.onLoad
      return;
    }

    setIsDetecting(true);
    setError(null);
    console.log("Detecting objects...");

    try {
      // Assert imageRef.current is not null as we checked above
      const predictions: cocoSsd.DetectedObject[] = await model.detect(imageRef.current as HTMLImageElement);
      setDetections(predictions as DetectedObject[]); // Cast to our extended type if necessary
      console.log("Detections:", predictions);
    } catch (err) {
      console.error("Error during detection:", err);
      setError("An error occurred during object detection.");
    } finally {
      setIsDetecting(false);
    }
  };

  const triggerFileInput = (): void => {
    fileInputRef.current?.click(); // Optional chaining
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Object Detector</h1>
        {isLoadingModel && <p>Loading AI Model... Please wait.</p>}
        {error && <p className="error-message">{error}</p>}

        {!isLoadingModel && !error && (
          <>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              ref={fileInputRef}
              style={{ display: 'none' }}
            />
            <button onClick={triggerFileInput} disabled={isDetecting || isLoadingModel}>
              Upload Image
            </button>
          </>
        )}
      </header>

      <main>
        {imageSrc && (
          <div className="image-container">
            <img
              ref={imageRef}
              src={imageSrc}
              alt="Uploaded preview"
              onLoad={detectObjects} // Detect when image is loaded
            />
          </div>
        )}

        {isDetecting && <p className="status-message">Detecting objects...</p>}

        {detections.length > 0 && (
          <div className="results">
          <h2>Detected Objects:</h2>
          <ul>
            {detections
              .filter(detection => detection.score >= MIN_CONFIDENCE) // Filter here
              .map((detection, index) => (
                <li key={index}>
                  {detection.class} (Confidence: {Math.round(detection.score * 100)}%)
                </li>
            ))}
          </ul>
          {detections.filter(d => d.score < MIN_CONFIDENCE).length > 0 && (
            <p style={{fontSize: '0.8em', color: 'gray'}}>
              ({detections.filter(d => d.score < MIN_CONFIDENCE).length} additional detections below {MIN_CONFIDENCE*100}% confidence not shown)
            </p>
          )}
        </div>
        )}
      </main>
    </div>
  );
}

export default App;