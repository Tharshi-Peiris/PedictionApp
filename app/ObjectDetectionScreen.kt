package com.example.prediction

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import android.widget.Toast
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
fun ObjectDetectionScreen(navController: NavHostController) {
    val context = LocalContext.current
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            imageBitmap = loadBitmapFromUri(context, it)
            resultImageBitmap = null
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
                text = "Leaf Area Detection",
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

            imageBitmap?.let { bitmap ->
                Text(
                    text = "Original Image",
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(250.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        if (!isLoading) {
                            isLoading = true
                            sendDetectionRequest(bitmap) { resultBitmap, error ->
                                isLoading = false
                                resultBitmap?.let {
                                    resultImageBitmap = it
                                } ?: Toast.makeText(context, "Error: $error", Toast.LENGTH_LONG).show()
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isLoading
                ) {
                    Text("Get Detection Result")
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
            }

            resultImageBitmap?.let {
                Text(
                    text = "Detection Result",
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Detection Result",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp)
                )
            }
        }
    }
}

// ========== Object Detection API ==========

fun sendDetectionRequest(
    bitmap: Bitmap,
    serverUrl: String = "http://192.168.8.101:5000/predict_detection",
    onResult: (Bitmap?, String?) -> Unit
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
            // Ensure callback runs on main thread
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                onResult(null, e.message)
            }
        }

        override fun onResponse(call: Call, response: Response) {
            response.use {
                val responseCode = response.code
                println("Detection response code: $responseCode")

                if (!response.isSuccessful) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        onResult(null, "Server error: $responseCode")
                    }
                    return
                }

                val responseBody = response.body?.string()
                println("Detection response body length: ${responseBody?.length}")

                if (responseBody != null) {
                    try {
                        val jsonObject = Gson().fromJson(responseBody, Map::class.java)
                        val base64Image = jsonObject["annotated_image"] as? String

                        println("Base64 image exists: ${base64Image != null}")
                        println("Base64 image length: ${base64Image?.length}")

                        if (base64Image != null) {
                            val imageBytes = Base64.decode(base64Image, Base64.DEFAULT)
                            val bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                            println("Decoded bitmap: ${bmp != null}")
                            println("Bitmap size: ${bmp?.width}x${bmp?.height}")

                            // Ensure callback runs on main thread
                            android.os.Handler(android.os.Looper.getMainLooper()).post {
                                onResult(bmp, null)
                            }
                        } else {
                            android.os.Handler(android.os.Looper.getMainLooper()).post {
                                onResult(null, "No annotated image found in response")
                            }
                        }
                    } catch (e: Exception) {
                        println("Detection parsing error: ${e.message}")
                        e.printStackTrace()
                        android.os.Handler(android.os.Looper.getMainLooper()).post {
                            onResult(null, "Parsing error: ${e.message}")
                        }
                    }
                } else {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        onResult(null, "Empty response")
                    }
                }
            }
        }
    })
}