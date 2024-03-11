import React, { useState } from 'react';
import Webcam from 'react-webcam';
import './styles.css';

const ASLInterface = () => {
  const [outputText, setOutputText] = useState('A');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const webcamRef = React.useRef(null);

  const handleSpeak = () => {
    // Implement text-to-speech logic here
    setIsSpeaking(true);
    // After speech, set setIsSpeaking(false);
  };

  return (
    <div className="container">
      {/* ASLRS text above webcam */}
      <p className="logo">ASLRS</p>

      {/* Webcam feed container */}
      <div className="webcam-container">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="w-full h-full rounded-lg"
        />
      </div>

      {/* Output text container */}
      <div className="output-container">
        <div className="output-text">
          <p className="text-4xl font-bold">Output: {outputText}</p>
        </div>

        {/* Audio icon for text-to-speech */}
        <button
          onClick={handleSpeak}
          disabled={isSpeaking}
          className="audio-icon"
        >
          {isSpeaking ? 'Speaking...' : '🔊'}
        </button>
      </div>
    </div>
  );
};

export default ASLInterface;
