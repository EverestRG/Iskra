package com.example.iskra

import android.app.Activity
import android.content.Context
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import android.content.SharedPreferences
import android.view.MotionEvent
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import android.graphics.Rect
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class SettingsActivity : AppCompatActivity() {

    private lateinit var ipEditText: EditText
    private lateinit var userIDText: EditText
    private lateinit var themeSwitch: Switch
    private lateinit var saveButton: Button
    private lateinit var resetButton: Button
    private lateinit var sharedPreferences: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        sharedPreferences = getSharedPreferences("Settings", MODE_PRIVATE)
        val isDarkTheme = sharedPreferences.getBoolean("darkTheme", false)
        setTheme(if (isDarkTheme) R.style.Theme_Dark else R.style.Theme_Light)

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        ipEditText = findViewById(R.id.ipEditText)
        themeSwitch = findViewById(R.id.themeSwitch)
        saveButton = findViewById(R.id.saveButton)
        resetButton = findViewById(R.id.resetButton)
        userIDText = findViewById(R.id.userIDText)

        // Загружаем сохранённые настройки
        val savedIp = sharedPreferences.getString("serverIp", "https://iskra-ai-server.loca.lt")
        ipEditText.setText(savedIp)
        userIDText.setText(sharedPreferences.getString("userID", ""))
        themeSwitch.isChecked = isDarkTheme

        var currentToast: Toast? = null

        fun showConfirmationDialog(context: Context, onResult: (Boolean) -> Unit) {
            // Проверяем, что контекст активности доступен
            if (!isFinishing) {  // Это проверка, что активность еще существует
                runOnUiThread {
                    val builder = AlertDialog.Builder(context, R.style.MyDialogStyle)
                    builder.setTitle("Confirmation")
                    builder.setMessage("Are you sure you want to erase all the data?")

                    builder.setPositiveButton("Yes") { dialog, _ ->
                        onResult(true)
                        dialog.dismiss()
                    }

                    builder.setNegativeButton("No") { dialog, _ ->
                        onResult(false)
                        dialog.dismiss()
                    }

                    builder.create().show()
                }
            } else {
                // Обрабатываем случай, когда контекст уже не доступен
                Toast.makeText(applicationContext, "Activity is no longer available", Toast.LENGTH_SHORT).show()
            }
        }

        resetButton.setOnClickListener {
            runOnUiThread {
                showConfirmationDialog(this) { result ->
                    if (result) {
                        val editor = sharedPreferences.edit()
                        editor.putString("existingText", "")
                        editor.apply()
                        val json = JSONObject()
                        json.put("prompt", "%%<|RESET|>%%")
                        json.put("user_id", sharedPreferences.getString("userID", "") ?: "")
                        val client = OkHttpClient.Builder()
                            .connectTimeout(10, TimeUnit.SECONDS)  // Тайм-аут соединения
                            .writeTimeout(60, TimeUnit.SECONDS)    // Тайм-аут записи
                            .readTimeout(60, TimeUnit.SECONDS)     // Тайм-аут чтения
                            .build()
                        val body = RequestBody.create(
                            "application/json; charset=utf-8".toMediaType(),
                            json.toString()
                        )
                        val request = Request.Builder()
                            .url(
                                "${sharedPreferences.getString("serverIp", "https://iskra-ai-server.loca.lt") ?: "https://iskra-ai-server.loca.lt" + "/get_response"}/get_response")
                            .post(body)
                            .build()
                        client.newCall(request).enqueue(object : Callback {
                            override fun onFailure(call: Call, e: IOException) {
                                runOnUiThread {
                                    currentToast?.cancel()
                                    currentToast = Toast.makeText(
                                        applicationContext,
                                        "Reset failure:\nFailed to connect to server!",
                                        Toast.LENGTH_SHORT
                                    )
                                    currentToast?.show()
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
                                        if (responseTextValue == "%|DONE|%") {
                                            currentToast?.cancel()
                                            currentToast = Toast.makeText(
                                                applicationContext,
                                                "Reset done",
                                                Toast.LENGTH_SHORT
                                            )
                                            currentToast?.show()
                                        }
                                    } catch (e: Exception) {
                                        currentToast?.cancel()
                                        currentToast = Toast.makeText(
                                            applicationContext,
                                            "Reset server error",
                                            Toast.LENGTH_SHORT
                                        )
                                        currentToast?.show()
                                    }
                                }
                            }
                        })
                        setResult(Activity.RESULT_OK)
                        finish()  // Закрываем экран настроек
                    }
                }
            }
        }

        // Сохраняем настройки
        saveButton.setOnClickListener {
            val editor = sharedPreferences.edit()
            editor.putString("serverIp", ipEditText.text.toString())
            editor.putString("userID", userIDText.text.toString())
            editor.putBoolean("darkTheme", themeSwitch.isChecked)
            editor.apply()

            setResult(Activity.RESULT_OK)

            finish()  // Закрываем экран настроек
        }
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.action == MotionEvent.ACTION_DOWN && currentFocus != null) {
            val focusedView = currentFocus
            if (focusedView is EditText) {
                val focusedRect = Rect()
                focusedView.getGlobalVisibleRect(focusedRect)

                // Проверяем, находится ли касание внутри поля ввода
                if (!focusedRect.contains(ev.rawX.toInt(), ev.rawY.toInt())) {
                    val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
                    imm.hideSoftInputFromWindow(focusedView.windowToken, 0)
                    focusedView.clearFocus()
                }
            }
        }
        return super.dispatchTouchEvent(ev)
    }
}
