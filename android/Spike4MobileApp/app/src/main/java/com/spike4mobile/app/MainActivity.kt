package com.spike4mobile.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var resultText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val runButton: Button = findViewById(R.id.runButton)
        resultText = findViewById(R.id.resultText)

        runButton.setOnClickListener {
            runInference()
        }
    }

    private fun runInference() {
        try {
            val modelPath = "snn_gesture_trained.onnx"
            val inputStream = assets.open(modelPath)
            val bytes = inputStream.readBytes()
            inputStream.close()

            val tempFile = File.createTempFile("snn_model_", ".onnx", cacheDir)
            tempFile.outputStream().use { stream -> stream.write(bytes) }

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(1)
            sessionOptions.setInterOpNumThreads(1)

            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(tempFile.absolutePath, sessionOptions)

            val inputShape = longArrayOf(1, 8, 2, 32, 32)
            val input = FloatArray(1 * 8 * 2 * 32 * 32) { 0.0f }
            val buffer = FloatBuffer.wrap(input)
            val tensor = OnnxTensor.createTensor(env, buffer, inputShape)
            val outputs = session.run(mapOf("event_sequence" to tensor))
            val result = outputs[0].value as FloatArray
            val logits = result.copyOf()

            val predicted = logits.withIndex().maxByOrNull { it.value }?.index ?: -1
            resultText.text = "Predicted class: $predicted\nFirst logits: ${logits.take(11).joinToString()}"
            Log.i("Spike4Mobile", "Inference completed: predicted=$predicted")
        } catch (throwable: Throwable) {
            Log.e("Spike4Mobile", "Inference failed", throwable)
            resultText.text = "ERROR: ${throwable.message}"
        }
    }
}
