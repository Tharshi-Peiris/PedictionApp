// PATCHED FILE: Updated Kotlin client to read "annotated_image" field instead of processing "mask"
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
import java.io.ByteArrayInputStream
import java.io.IOException
import java.util.concurrent.TimeUnit
import androidx.compose.ui.graphics.Color

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SegmentationScreen(navController: NavHostController) {
    val context = LocalContext.current
    var originalImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var segmentedImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var segmentationInfo by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            originalImageBitmap = loadBitmapFromUri(context, it)
            segmentedImageBitmap = null
            segmentationInfo = null
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
                "Affected Area Highlight",
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

            originalImageBitmap?.let { bitmap ->
                Text("Original Image", style = MaterialTheme.typography.titleMedium)
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = {
                    if (!isLoading) {
                        isLoading = true
                        sendSegmentationRequest(context, bitmap) { resultBitmap, info, error ->
                            isLoading = false
                            if (error != null) {
                                Toast.makeText(context, "Error: $error", Toast.LENGTH_LONG).show()
                            } else {
                                segmentedImageBitmap = resultBitmap
                                segmentationInfo = info
                            }
                        }
                    }
                }, enabled = !isLoading, modifier = Modifier.fillMaxWidth()) {
                    Text("Get Segmentation Result")
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            if (isLoading) {
                CircularProgressIndicator()
                Text(
                    "Processing segmentation...",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }

            segmentedImageBitmap?.let { bitmap ->
                Text("Segmented Image", style = MaterialTheme.typography.titleMedium)
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                segmentationInfo?.let { info ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {

                        
                    }
                }
            }
        }
    }
}

fun sendSegmentationRequest(
    context: android.content.Context,
    bitmap: Bitmap,
    serverUrl: String = "http://192.168.8.101:5000/predict_segmentation",
    onResult: (Bitmap?, String?, String?) -> Unit
) {
    val base64Image = bitmapToBase64WithDataUrl(bitmap)
    val json = """{ "image": "$base64Image" }"""

    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    val request = Request.Builder()
        .url(serverUrl)
        .post(RequestBody.create("application/json".toMediaType(), json))
        .addHeader("Content-Type", "application/json")
        .build()

    client.newCall(request).enqueue(object : Callback {
        override fun onFailure(call: Call, e: IOException) {
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                onResult(null, null, "Request failed: ${e.message}")
            }
        }

        override fun onResponse(call: Call, response: Response) {
            response.use {
                val responseBody = response.body?.string()
                if (!response.isSuccessful || responseBody == null) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        onResult(null, null, "Server error: ${response.code}")
                    }
                    return
                }

                try {
                    val jsonMap = Gson().fromJson(responseBody, Map::class.java)
                    val success = jsonMap["success"] as? Boolean ?: false
                    val annotatedImage = jsonMap["annotated_image"] as? String
                    val detections = jsonMap["detections"] as? List<*>

                    if (success && !annotatedImage.isNullOrBlank()) {
                        val bytes = Base64.decode(annotatedImage, Base64.DEFAULT)
                        val bitmap = BitmapFactory.decodeStream(ByteArrayInputStream(bytes))

                        val info = buildString {
                            append("✅ Segmentation successful\n")
                            append("Annotated image size: ${annotatedImage.length} characters\n")
                            append("Original image: ${bitmap.width}x${bitmap.height}\n")

                            if (!detections.isNullOrEmpty()) {
                                append("Detections found: ${detections.size}\n\n")
                                detections.forEachIndexed { index, detection ->
                                    append("Detection ${index + 1}:\n")
                                    if (detection is Map<*, *>) {
                                        detection.forEach { (k, v) -> append("  $k: $v\n") }
                                    }
                                }
                            } else {
                                append("No specific detections returned.")
                            }
                        }

                        android.os.Handler(android.os.Looper.getMainLooper()).post {
                            onResult(bitmap, info, null)
                        }
                    } else {
                        android.os.Handler(android.os.Looper.getMainLooper()).post {
                            onResult(null, null, "Segmentation failed or missing annotated image")
                        }
                    }
                } catch (e: Exception) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        onResult(null, null, "Parsing error: ${e.message}")
                    }
                }
            }
        }
    })
}