package com.example.prediction

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.prediction.ui.theme.PredictionTheme
import java.io.ByteArrayOutputStream
import androidx.compose.foundation.background
import androidx.compose.ui.graphics.Color
import kotlinx.coroutines.delay
import androidx.compose.foundation.Image
import androidx.compose.ui.res.painterResource


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PredictionTheme {
                val navController = rememberNavController()
                NavHost(
                    navController = navController,
                    startDestination = "splash"          
                ) {
                    composable("splash")         { SplashScreen(navController) }
                    composable("main")           { MainScreen(navController) }
                    composable("classification") { ClassificationScreen(navController) }
                    composable("object_detection") { ObjectDetectionScreen(navController) }
                    composable("segmentation")   { SegmentationScreen(navController) }
                }
            }
        }

    }
}

@Composable
fun MainScreen(navController: NavHostController) {
    val leafGreen = Color(0xFFCEF7D1)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(leafGreen)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // 🌿 Larger image and positioned a bit higher
        Image(
            painter = painterResource(id = R.drawable.plant), // Use your actual image
            contentDescription = "LeafCare Logo",
            modifier = Modifier
                .size(250.dp) // Increased size
                .padding(bottom = 8.dp) // Less bottom padding (was 32.dp)
        )

        Button(
            onClick = { navController.navigate("classification") },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Text("Check Leaf Health", style = MaterialTheme.typography.titleMedium)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { navController.navigate("object_detection") },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Text("Find Problem Leaves", style = MaterialTheme.typography.titleMedium)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { navController.navigate("segmentation") },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Text("Show Damage Area", style = MaterialTheme.typography.titleMedium)
        }
    }
}

// ========== Shared Utilities ==========

fun loadBitmapFromUri(context: android.content.Context, uri: Uri): Bitmap? {
    return try {
        if (Build.VERSION.SDK_INT < 28) {
            @Suppress("DEPRECATION")
            android.provider.MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        } else {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

fun bitmapToBase64(bitmap: Bitmap): String {
    val outputStream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
    val byteArray = outputStream.toByteArray()
    val base64String = Base64.encodeToString(byteArray, Base64.NO_WRAP)
    return base64String
}

fun bitmapToBase64WithDataUrl(bitmap: Bitmap): String {
    val outputStream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
    val byteArray = outputStream.toByteArray()
    val base64String = Base64.encodeToString(byteArray, Base64.NO_WRAP)
    return "data:image/jpeg;base64,$base64String"
}

@Composable
fun SplashScreen(navController: NavHostController) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFCEF7D1)), // leaf-green background
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "LeafCare",
                style = MaterialTheme.typography.headlineLarge
            )

            Spacer(modifier = Modifier.height(16.dp)) // spacing between text and image

            Image(
                painter = painterResource(id = R.drawable.leaf), // replace with your actual image name
                contentDescription = "LeafCare Logo",
                modifier = Modifier.size(120.dp) // adjust size as needed
            )
        }
    }

    LaunchedEffect(Unit) {
        delay(2000) // 2-second splash
        navController.navigate("main") {
            popUpTo("splash") { inclusive = true }
        }
    }
}


fun Double.format(digits: Int) = "%.${digits}f".format(this)