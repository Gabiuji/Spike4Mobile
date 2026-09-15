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
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
            val modelName = "snn_gesture_trained.onnx"
            val modelFile = File(cacheDir, modelName)
            assets.open(modelName).use { input ->
                modelFile.outputStream().use { output -> input.copyTo(output) }
            }

            val externalDataName = "$modelName.data"
            val externalDataFile = File(cacheDir, externalDataName)
            assets.open(externalDataName).use { input ->
                externalDataFile.outputStream().use { output -> input.copyTo(output) }
            }

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(1)
            sessionOptions.setInterOpNumThreads(1)

            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(modelFile.absolutePath, sessionOptions)

            val inputShape = longArrayOf(1, 8, 2, 32, 32)
            val sampleBytes = assets.open("android_input.bin").use { it.readBytes() }
            val inputBuffer = ByteBuffer.wrap(sampleBytes)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            val input = FloatArray(inputBuffer.remaining())
            inputBuffer.get(input)
            val buffer = FloatBuffer.wrap(input)
            val tensor = OnnxTensor.createTensor(env, buffer, inputShape)
            val inferenceStart = System.nanoTime()
            val outputs = session.run(mapOf("event_sequence" to tensor))
            val inferenceMs = (System.nanoTime() - inferenceStart) / 1_000_000.0
            val result = outputs[0].value as Array<FloatArray>
            val logits = result.firstOrNull() ?: floatArrayOf()

            val predicted = logits.withIndex().maxByOrNull { it.value }?.index ?: -1
            resultText.text = "Predicted class: $predicted (expected: 0)\nInference: %.2f ms\nFirst logits: %s".format(
                inferenceMs,
                logits.take(11).joinToString()
            )
            Log.i("Spike4Mobile", "Inference completed: predicted=$predicted latency_ms=$inferenceMs")
        } catch (throwable: Throwable) {
            Log.e("Spike4Mobile", "Inference failed", throwable)
            resultText.text = "ERROR: ${throwable.message}"
        }
    }
}
