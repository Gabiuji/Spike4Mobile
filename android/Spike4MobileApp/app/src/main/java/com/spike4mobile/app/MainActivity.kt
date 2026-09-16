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
        val runRawButton: Button = findViewById(R.id.runRawButton)
        resultText = findViewById(R.id.resultText)

        runButton.setOnClickListener {
            runInference()
        }
        runRawButton.setOnClickListener {
            runRawSample()
        }
    }

    private fun runRawSample() {
        try {
            val rawBytes = assets.open("android_raw_sample.bin").use { it.readBytes() }
            val raw = ByteBuffer.wrap(rawBytes).order(ByteOrder.LITTLE_ENDIAN)
            val windowStart = raw.long
            val windowEnd = raw.long
            val eventCount = raw.int
            val bins = 8
            val width = 32
            val height = 32
            val grid = FloatArray(bins * 2 * height * width)
            val duration = (windowEnd - windowStart).toDouble()
            repeat(eventCount) {
                val x = raw.int / 4
                val y = raw.int / 4
                val timestamp = raw.long
                val polarity = raw.get().toInt()
                if (x !in 0 until width || y !in 0 until height) return@repeat
                val normalizedTime = (timestamp - windowStart).toDouble() / duration * (bins - 1)
                val lowerBin = minOf(normalizedTime.toInt(), bins - 1)
                val upperWeight = normalizedTime - lowerBin
                val channel = if (polarity > 0) 1 else 0
                val lowerIndex = ((lowerBin * 2 + channel) * height + y) * width + x
                grid[lowerIndex] += (1.0 - upperWeight).toFloat()
                if (lowerBin + 1 < bins) {
                    val upperIndex = (((lowerBin + 1) * 2 + channel) * height + y) * width + x
                    grid[upperIndex] += upperWeight.toFloat()
                }
            }

            val mass = grid.sum().coerceAtLeast(1.0f)
            for (index in grid.indices) grid[index] = grid[index] / mass * 1000.0f
            val expectedBytes = assets.open("android_raw_expected.bin").use { it.readBytes() }
            val expected = ByteBuffer.wrap(expectedBytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
            var maxError = 0.0f
            for (index in grid.indices) maxError = maxOf(maxError, kotlin.math.abs(grid[index] - expected.get(index)))
            val predicted = runModel(FloatBuffer.wrap(grid))
            resultText.text = "Raw sample: predicted=$predicted (expected: 0)\nEvents: $eventCount\nVoxel max error: %.6f".format(maxError)
            Log.i("Spike4Mobile", "Raw sample completed: events=$eventCount predicted=$predicted voxel_max_error=$maxError")
        } catch (throwable: Throwable) {
            Log.e("Spike4Mobile", "Raw sample failed", throwable)
            resultText.text = "ERROR: ${throwable.message}"
        }
    }

    private fun runModel(buffer: FloatBuffer): Int {
        val modelName = "snn_gesture_trained.onnx"
        val modelFile = File(cacheDir, modelName)
        assets.open(modelName).use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }
        assets.open("$modelName.data").use { input ->
            File(cacheDir, "$modelName.data").outputStream().use { output -> input.copyTo(output) }
        }

        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setIntraOpNumThreads(1)
        sessionOptions.setInterOpNumThreads(1)
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(modelFile.absolutePath, sessionOptions)
        val tensor = OnnxTensor.createTensor(env, buffer, longArrayOf(1, 8, 2, 32, 32))
        val outputs = session.run(mapOf("event_sequence" to tensor))
        val result = outputs[0].value as Array<FloatArray>
        val logits = result.firstOrNull() ?: floatArrayOf()
        val predicted = logits.withIndex().maxByOrNull { it.value }?.index ?: -1
        tensor.close()
        outputs.close()
        session.close()
        sessionOptions.close()
        return predicted
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
            val inputBytes = assets.open("android_test_inputs.bin").use { it.readBytes() }
            val labelBytes = assets.open("android_test_labels.bin").use { it.readBytes() }
            val inputBuffer = ByteBuffer.wrap(inputBytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val labelBuffer = ByteBuffer.wrap(labelBytes).order(ByteOrder.nativeOrder()).asIntBuffer()
            val sampleSize = 8 * 2 * 32 * 32
            val sampleCount = labelBuffer.remaining()
            var correct = 0
            var totalInferenceMs = 0.0
            val predictions = mutableListOf<Int>()

            repeat(sampleCount) {
                val input = FloatArray(sampleSize)
                inputBuffer.get(input)
                val buffer = FloatBuffer.wrap(input)
                val tensor = OnnxTensor.createTensor(env, buffer, inputShape)
                val inferenceStart = System.nanoTime()
                val outputs = session.run(mapOf("event_sequence" to tensor))
                totalInferenceMs += (System.nanoTime() - inferenceStart) / 1_000_000.0
                val result = outputs[0].value as Array<FloatArray>
                val logits = result.firstOrNull() ?: floatArrayOf()
                val predicted = logits.withIndex().maxByOrNull { it.value }?.index ?: -1
                predictions += predicted
                if (predicted == labelBuffer.get(it)) correct++
                tensor.close()
                outputs.close()
            }

            val accuracy = correct.toDouble() / sampleCount
            val meanInferenceMs = totalInferenceMs / sampleCount
            resultText.text = "Android test: $correct/$sampleCount (%.1f%%)\nMean inference: %.2f ms\nPredictions: %s".format(
                accuracy * 100.0,
                meanInferenceMs,
                predictions.joinToString()
            )
            Log.i("Spike4Mobile", "Batch completed: correct=$correct total=$sampleCount accuracy=$accuracy mean_latency_ms=$meanInferenceMs")
        } catch (throwable: Throwable) {
            Log.e("Spike4Mobile", "Inference failed", throwable)
            resultText.text = "ERROR: ${throwable.message}"
        }
    }
}
