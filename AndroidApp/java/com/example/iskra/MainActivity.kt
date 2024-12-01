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
import java.util.UUID
import android.content.Context
import android.graphics.Rect
import android.text.Editable
import android.text.TextWatcher
import android.view.inputmethod.InputMethodManager
import android.widget.ScrollView

class MainActivity : AppCompatActivity() {

    private lateinit var inputText: EditText
    private lateinit var sendRequestButton: Button
    private lateinit var responseText: TextView
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var darkThemeSwitch: Switch
    private lateinit var scrollView: ScrollView

    private lateinit var serverIp: String

    override fun onCreate(savedInstanceState: Bundle?) {
        // Загружаем настройки
        sharedPreferences = getSharedPreferences("Settings", MODE_PRIVATE)
        val isDarkTheme = sharedPreferences.getBoolean("darkTheme", false)
        setTheme(if (isDarkTheme) R.style.Theme_Dark else R.style.Theme_Light)
        val uid = sharedPreferences.getString("userID", "") ?: ""
        if (uid == "") {
            val editorr = sharedPreferences.edit()
            editorr.putString("userID", UUID.randomUUID().toString())
            editorr.apply()
        }

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var currentToast: Toast? = null
        var lastrequest = ""

        // Инициализация элементов
        inputText = findViewById(R.id.inputText)
        sendRequestButton = findViewById(R.id.sendRequestButton)
        responseText = findViewById(R.id.responseText)
        scrollView = findViewById(R.id.scrollView)
        var settingsButton: Button = findViewById(R.id.settingsButton)

        inputText.setText(sharedPreferences.getString("inpTxt", "") ?: "")

        try {
            requestHistory()
        } catch (e: Exception) {
            null
        }

        // Получаем сохранённый IP-адрес из SharedPreferences
        serverIp = sharedPreferences.getString("serverIp", "https://iskra-ai-server.loca.lt") ?: "https://iskra-ai-server.loca.lt"

        // Регистрация ActivityResultLauncher
        val settingsLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                // Перезапуск текущей активности после успешного завершения другой активности
                recreate()
            }
        }

        scrollView.viewTreeObserver.addOnGlobalLayoutListener {
            val rect = Rect()
            scrollView.getWindowVisibleDisplayFrame(rect)

            val screenHeight = scrollView.rootView.height
            val keypadHeight = screenHeight - rect.bottom

            if (keypadHeight > screenHeight * 0.15) { // Условие: клавиатура открыта
                scrollView.post {
                    scrollView.smoothScrollTo(0, scrollView.bottom)
                }
            }
        }

        settingsButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            settingsLauncher.launch(intent)
            val editor = sharedPreferences.edit()
            editor.putString("existingText", responseText.text.toString())
            editor.putString("inpTxt", inputText.text.toString())
            editor.apply()
        }

        inputText.addTextChangedListener(object: TextWatcher{
            override fun beforeTextChanged(
                charSequence: CharSequence?,
                start: Int,
                before: Int,
                count: Int
            ) {
                // Действие до изменения текста (например, удаление текста)
            }
            override fun onTextChanged(
                charSequence: CharSequence?,
                start: Int,
                before: Int,
                count: Int
            ) {
                if (inputText.text.toString() != ""){
                    sendRequestButton.text = "▶"
                }
            }
            override fun afterTextChanged(editable: Editable?) {
                // Действие после изменения текста
                // Здесь можно обработать изменённый текст
                editable?.let {
                    null
                }
            }
        })

        // Кнопка для отправки запроса
        sendRequestButton.setOnClickListener {
            if (sendRequestButton.text != "↻") {
                lastrequest = inputText.text.toString()
            }
            if (currentFocus != null) {
                val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
                imm.hideSoftInputFromWindow(currentFocus!!.windowToken, 0)
            }
            if (lastrequest.isNotBlank()) {
                sendRequest(lastrequest)
            } else {
                currentToast?.cancel()
                currentToast = Toast.makeText(this, "Please enter a prompt.", Toast.LENGTH_SHORT)
                currentToast?.show()
            }
        }

        // Проверка и запрос разрешений
        //checkPermissions()
    }

    private fun requestHistory() {
        var currentToasts: Toast? = null
        try {
            val json = JSONObject()
            json.put("user_id", sharedPreferences.getString("userID", "") ?: "")
            serverIp =
                sharedPreferences.getString("serverIp", "https://iskra-ai-server.loca.lt")
                    ?: "https://iskra-ai-server.loca.lt"
            val client = OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)  // Тайм-аут соединения
                .writeTimeout(60, TimeUnit.SECONDS)    // Тайм-аут записи
                .readTimeout(60, TimeUnit.SECONDS)     // Тайм-аут чтения
                .build()
            val body =
                RequestBody.create("application/json; charset=utf-8".toMediaType(), json.toString())
            val request = Request.Builder()
                .url(serverIp + "/get_history")
                .post(body)
                .build()
            var currentToastt: Toast? = null
            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    runOnUiThread {
                        currentToastt?.cancel()
                        currentToastt = Toast.makeText(
                            applicationContext,
                            "Failed requesting dialog history from server!",
                            Toast.LENGTH_SHORT
                        )
                        currentToastt?.show()
                        var txts = sharedPreferences.getString("existingText", "") ?: ""
                        txts = txts.replace("[You]", "<font color='#27AE60'>[You]</font>")
                        txts = txts.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                        txts = txts.replace("\n", "<br>")
                        responseText.text =
                            HtmlCompat.fromHtml(txts, HtmlCompat.FROM_HTML_MODE_LEGACY)
                        scrollToBottom()
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    val responseString = response.body?.string()
                    runOnUiThread {
                        try {
                            //Парсим JSON
                            val jsonResponse =
                                JSONObject(responseString ?: "No response from server.")
                            var responseTextValue = jsonResponse.getString("response")
                            responseTextValue = responseTextValue.replace(
                                "[You]",
                                "<font color='#27AE60'>[You]</font>"
                            )
                            responseTextValue = responseTextValue.replace(
                                "[Iskra]",
                                "<font color='#27AE60'>[Iskra]</font>"
                            )
                            responseTextValue = responseTextValue.replace("\n", "<br>")
                            responseText.text = HtmlCompat.fromHtml(
                                responseTextValue,
                                HtmlCompat.FROM_HTML_MODE_LEGACY
                            )
                        } catch (e: Exception) {
                            currentToastt?.cancel()
                            currentToastt = Toast.makeText(
                                applicationContext,
                                "Failed parsing dialog history from server!",
                                Toast.LENGTH_SHORT
                            )
                            currentToastt?.show()
                            var txts = sharedPreferences.getString("existingText", "") ?: ""
                            txts = txts.replace("[You]", "<font color='#27AE60'>[You]</font>")
                            txts = txts.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                            txts = txts.replace("\n", "<br>")
                            responseText.text =
                                HtmlCompat.fromHtml(txts, HtmlCompat.FROM_HTML_MODE_LEGACY)
                        }
                        val editor = sharedPreferences.edit()
                        editor.putString("existingText", responseText.text.toString())
                        editor.apply()
                        scrollToBottom()
                    }
                }
            })
        } catch (e: Exception){
            currentToasts?.cancel()
            currentToasts = Toast.makeText(
                applicationContext,
                "Failed pinging server!",
                Toast.LENGTH_SHORT
            )
            currentToasts?.show()
        }
    }

    // Отправка запроса на сервер
    private fun sendRequest(prompt: String) {
        sendRequestButton.isEnabled = false
        sendRequestButton.text = "..."
        var txt = responseText.text.toString() + "<br><br><font color='#27AE60'>[You]</font>: ${prompt}<br><br><font color='#F4D03F'>Generating...</font>"
        txt = txt.replace("[You]", "<font color='#27AE60'>[You]</font>")
        txt = txt.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
        txt = txt.replace("\n", "<br>")
        responseText.text = HtmlCompat.fromHtml(txt, HtmlCompat.FROM_HTML_MODE_LEGACY)
        inputText.setText("")
        val json = JSONObject()
        json.put("prompt", prompt)
        json.put("user_id", sharedPreferences.getString("userID", "") ?: "")
        serverIp = sharedPreferences.getString("serverIp", "https://iskra-ai-server.loca.lt") ?: "https://iskra-ai-server.loca.lt"
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
        var retrying = false
        var currentToastts: Toast? = null
        try {
            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    runOnUiThread {
                        var teeext = responseText.text.toString().replace(
                            "Generating...",
                            "<font color='#A93226'>Failed to connect to server: ${e.message}</font>"
                        )
                        teeext = teeext.replace("[You]", "<font color='#27AE60'>[You]</font>")
                        teeext = teeext.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                        teeext = teeext.replace("\n", "<br>")
                        responseText.text =
                            HtmlCompat.fromHtml(teeext, HtmlCompat.FROM_HTML_MODE_LEGACY)
                        retrying = true
                        scrollToBottom()
                    }
                    sendRequestButton.isEnabled = true
                    if (retrying) {
                        sendRequestButton.text = "↻"
                    } else {
                        sendRequestButton.text = "▶"
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    val responseString = response.body?.string()
                    runOnUiThread {
                        try {
                            //Парсим JSON
                            val jsonResponse =
                                JSONObject(responseString ?: "No response from server.")
                            var responseTextValue = jsonResponse.getString("response")
                            responseTextValue = responseTextValue.replace(
                                "[You]",
                                "<font color='#27AE60'>[You]</font>"
                            )
                            responseTextValue = responseTextValue.replace(
                                "[Iskra]",
                                "<font color='#27AE60'>[Iskra]</font>"
                            )
                            responseTextValue = responseTextValue.replace("\n", "<br>")
                            responseText.text = HtmlCompat.fromHtml(
                                responseTextValue,
                                HtmlCompat.FROM_HTML_MODE_LEGACY
                            )
                        } catch (e: Exception) {
                            var teext = responseText.text.toString().replace(
                                "Generating...",
                                "<font color='#A93226'>Error parsing response.</font>"
                            )
                            teext = teext.replace("[You]", "<font color='#27AE60'>[You]</font>")
                            teext = teext.replace("[Iskra]", "<font color='#27AE60'>[Iskra]</font>")
                            teext = teext.replace("\n", "<br>")
                            responseText.text =
                                HtmlCompat.fromHtml(teext, HtmlCompat.FROM_HTML_MODE_LEGACY)
                            retrying = true
                        }
                        sendRequestButton.isEnabled = true
                        if (retrying) {
                            sendRequestButton.text = "↻"
                        } else {
                            sendRequestButton.text = "▶"
                        }
                        val editor = sharedPreferences.edit()
                        editor.putString("existingText", responseText.text.toString())
                        editor.apply()
                        scrollToBottom()
                    }
                }
            })
        } catch (e: Exception) {
            currentToastts?.cancel()
            currentToastts = Toast.makeText(applicationContext, "Server pinging error", Toast.LENGTH_SHORT)
            currentToastts?.show()
            sendRequestButton.isEnabled = true
            sendRequestButton.text = "▶"
        }
    }

    fun scrollToBottom() {
        if (responseText.text != "") {
            scrollView.post {
                scrollView.smoothScrollTo(0, scrollView.bottom)
            }
        }
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
