plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.spike4mobile.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.spike4mobile.app"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}

tasks.register<Copy>("copyOnnxModel") {
    val modelFile = file("${rootProject.projectDir}/../../host/artifacts/snn_gesture_trained.onnx")
    val sampleFile = file("${rootProject.projectDir}/../../host/artifacts/android_input.bin")
    if (modelFile.exists()) {
        from(modelFile.parentFile) {
            include(modelFile.name)
            include("${modelFile.name}.data")
            if (sampleFile.exists()) {
                include(sampleFile.name)
            }
        }
        into(file("${projectDir}/src/main/assets"))
    } else {
        println("WARN: ONNX model not found at ${modelFile.absolutePath}")
    }
}

tasks.named("preBuild") {
    dependsOn("copyOnnxModel")
}
