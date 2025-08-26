package com.example.prediction

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.google.gson.Gson
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import java.io.IOException
import androidx.compose.ui.graphics.Color

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClassificationScreen(navController: NavHostController) {
    val context = LocalContext.current
    var selectedImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var predictionResult by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedImageBitmap = loadBitmapFromUri(context, it)
            predictionResult = null
        }
    }

    Scaffold(
        bottomBar = {
            BottomAppBar(
                containerColor = Color.Transparent,
                tonalElevation = 0.dp
            ) {
                Button(
                    onClick = { navController.popBackStack() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                ) {
                    Text("← Back")
                }
            }
        }

    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Leaf Health Diagnosis",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            Button(
                onClick = { launcher.launch("image/*") },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Pick Image from Gallery")
            }

            Spacer(modifier = Modifier.height(16.dp))

            selectedImageBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        if (!isLoading) {
                            isLoading = true
                            sendPredictionRequest(bitmap) { prediction, error ->
                                isLoading = false
                                predictionResult = error ?: prediction
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isLoading
                ) {
                    Text("Get Classification Result")
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            if (isLoading) {
                CircularProgressIndicator()
                Text(
                    text = "Processing...",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp)
                )
            } else {
                predictionResult?.let {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Text(
                            text = "Result:",
                            style = MaterialTheme.typography.titleMedium,
                            modifier = Modifier.padding(16.dp, 16.dp, 16.dp, 8.dp)
                        )
                        Text(
                            text = it,
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(16.dp, 0.dp, 16.dp, 16.dp)
                        )
                    }
                }
            }
        }
    }
}

// ========== Classification API ==========

data class PredictionResponse(
    val success: Boolean,
    val prediction: PredictionDetail?
)

data class PredictionDetail(
    val predicted_class: String,
    val predicted_class_index: Int,
    val confidence: Double,
    val all_probabilities: Map<String, Double>
)

fun sendPredictionRequest(
    bitmap: Bitmap,
    serverUrl: String = "http://192.168.8.101:5000/predict",
    onResult: (String?, String?) -> Unit
) {
    val base64Image = bitmapToBase64(bitmap)
    val json = """{ "image": "$base64Image" }""".trimIndent()

    val client = OkHttpClient()
    val mediaType = "application/json".toMediaType()
    val body = RequestBody.create(mediaType, json)

    val request = Request.Builder()
        .url(serverUrl)
        .post(body)
        .addHeader("Content-Type", "application/json")
        .build()

    client.newCall(request).enqueue(object : Callback {
        override fun onFailure(call: Call, e: IOException) {
            onResult(null, e.message)
        }

        override fun onResponse(call: Call, response: Response) {
            response.use {
                if (!response.isSuccessful) {
                    onResult(null, "Server error: ${response.code}")
                    return
                }

                val responseBody = response.body?.string()
                if (responseBody != null) {
                    try {
                        val predictionResponse =
                            Gson().fromJson(responseBody, PredictionResponse::class.java)
                        if (predictionResponse.success && predictionResponse.prediction != null) {
                            val pred = predictionResponse.prediction
                            val resultText = pred.predicted_class

//                            val resultText = buildString {
//                                append("Class: ${pred.predicted_class}\n")
//                                append("Confidence: ${(pred.confidence * 100).format(2)}%\n")
//                                append("Probabilities:\n")
//                                pred.all_probabilities.forEach { (label, prob) ->
//                                    append("  $label: ${(prob * 100).format(2)}%\n")
//                                }
//                            }
                            onResult(resultText, null)
                        } else {
                            onResult("Prediction failed or empty", null)
                        }
                    } catch (e: Exception) {
                        onResult(null, "Parsing error: ${e.message}")
                    }
                } else {
                    onResult(null, "Empty response")
                }
            }
        }
    })
}