package com.example.iskra

import android.app.Activity
import android.Manifest
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import okhttp3.*
import org.json.JSONObject
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaType
import android.content.Intent
import android.content.pm.PackageManager
import java.util.concurrent.TimeUnit
import androidx.activity.result.contract.ActivityResultContracts
import android.widget.Toast
import androidx.core.text.HtmlCompat

class MainActivity : AppCompatActivity() {

    private lateinit var inputText: EditText
    private lateinit var sendRequestButton: Button
    private lateinit var responseText: TextView
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var darkThemeSwitch: Switch

    private lateinit var serverIp: String

    override fun onCreate(savedInstanceState: Bundle?) {
        // Загружаем настройки
        sharedPreferences = getSharedPreferences("Settings", MODE_PRIVATE)
        val isDarkTheme = sharedPreferences.getBoolean("darkTheme", false)
        setTheme(if (isDarkTheme) R.style.Theme_Dark else R.style.Theme_Light)

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var currentToast: Toast? = null

        // Инициализация элементов
        inputText = findViewById(R.id.inputText)
        sendRequestButton = findViewById(R.id.sendRequestButton)
        responseText = findViewById(R.id.responseText)

        // Получаем сохранённый IP-адрес из SharedPreferences
        serverIp = sharedPreferences.getString("serverIp", "https://eb93-77-222-113-143.ngrok-free.app") ?: "https://eb93-77-222-113-143.ngrok-free.app"

        // Регистрация ActivityResultLauncher
        val settingsLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                // Перезапуск текущей активности после успешного завершения другой активности
                recreate()
            }
        }

        val settingsButton: Button = findViewById(R.id.settingsButton)

        settingsButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            settingsLauncher.launch(intent)
        }

        // Кнопка для отправки запроса
        sendRequestButton.setOnClickListener {
            val prompt = inputText.text.toString()
            if (prompt.isNotBlank()) {
                sendRequest(prompt)
            } else {
                currentToast?.cancel()
                currentToast = Toast.makeText(this, "Please enter a prompt.", Toast.LENGTH_SHORT)
                currentToast?.show()
            }
        }

        // Проверка и запрос разрешений
        checkPermissions()
    }

    fun replaceHtmlString(input: String, oldText: String, newText: String): String {
        // Преобразуем HTML в Spannable
        var spannableText = HtmlCompat.fromHtml(input, HtmlCompat.FROM_HTML_MODE_LEGACY)

        // Преобразуем Spannable обратно в строку, заменяя старое на новое
        val result = spannableText.toString().replace(oldText, newText)

        //spannableText = HtmlCompat.fromHtml(result, HtmlCompat.FROM_HTML_MODE_LEGACY)

        // Возвращаем результат в виде HTML с сохранением тегов
        return spannableText.toString()//HtmlCompat.toHtml(spannableText, HtmlCompat.TO_HTML_PARAGRAPH_LINES_CONSECUTIVE)
    }

    // Отправка запроса на сервер
    private fun sendRequest(prompt: String) {
        sendRequestButton.isEnabled = false
        sendRequestButton.text = "..."
        var txt = responseText.text.toString() + "<br><br><font color='#27AE60'>[You]</font>: ${inputText.text}<br><br><font color='#F4D03F'>Generating...</font>"
        txt = txt.replace("[You]", "<font color='#27AE60'>[You]</font>")
        txt = txt.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
        txt = txt.replace("\n", "<br>")
        responseText.text = HtmlCompat.fromHtml(txt, HtmlCompat.FROM_HTML_MODE_LEGACY)
        inputText.setText("")
        val json = JSONObject()
        json.put("prompt", prompt)
        serverIp = sharedPreferences.getString("serverIp", "https://eb93-77-222-113-143.ngrok-free.app") ?: "https://eb93-77-222-113-143.ngrok-free.app"
        val client = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)  // Тайм-аут соединения
            .writeTimeout(60, TimeUnit.SECONDS)    // Тайм-аут записи
            .readTimeout(60, TimeUnit.SECONDS)     // Тайм-аут чтения
            .build()
        val body = RequestBody.create("application/json; charset=utf-8".toMediaType(), json.toString())
        val request = Request.Builder()
            .url(serverIp + "/get_response")
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    var teeext = responseText.text.toString().replace("Generating...", "<font color='#A93226'>Failed to connect to server: ${e.message}</font>")
                    teeext = teeext.replace("[You]", "<font color='#27AE60'>[You]</font>")
                    teeext = teeext.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                    teeext = teeext.replace("\n", "<br>")
                    responseText.text = HtmlCompat.fromHtml(teeext, HtmlCompat.FROM_HTML_MODE_LEGACY)
                }
                sendRequestButton.isEnabled = true
                sendRequestButton.text = "▶"
            }

            override fun onResponse(call: Call, response: Response) {
                val responseString = response.body?.string()
                runOnUiThread {
                    try {
                        //Парсим JSON
                        val jsonResponse = JSONObject(responseString ?: "No response from server.")
                        var responseTextValue = jsonResponse.getString("response")
                        responseTextValue = responseTextValue.replace("[You]", "<font color='#27AE60'>[You]</font>")
                        responseTextValue = responseTextValue.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                        responseTextValue = responseTextValue.replace("\n", "<br>")
                        responseText.text = HtmlCompat.fromHtml(responseTextValue, HtmlCompat.FROM_HTML_MODE_LEGACY)
                    } catch (e: Exception) {
                        var teext = responseText.text.toString().replace("Generating...", "<font color='#A93226'>Error parsing response.</font>")
                        teext = teext.replace("[You]", "<font color='#27AE60'>[You]</font>")
                        teext = teext.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                        teext = teext.replace("\n", "<br>")
                        responseText.text = HtmlCompat.fromHtml(teext, HtmlCompat.FROM_HTML_MODE_LEGACY)
                    }
                    sendRequestButton.isEnabled = true
                    sendRequestButton.text = "▶"
                }
            }
        })
    }

    // Проверка разрешений на доступ к данным устройства
    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.INTERNET
        )

        val missingPermissions = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missingPermissions.toTypedArray(), 1)
        }
    }
}
