package com.example.iskra

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import android.content.SharedPreferences

class SettingsActivity : AppCompatActivity() {

    private lateinit var ipEditText: EditText
    private lateinit var themeSwitch: Switch
    private lateinit var saveButton: Button
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

        // Загружаем сохранённые настройки
        val savedIp = sharedPreferences.getString("serverIp", "https://eb93-77-222-113-143.ngrok-free.app")

        ipEditText.setText(savedIp)
        themeSwitch.isChecked = isDarkTheme

        // Сохраняем настройки
        saveButton.setOnClickListener {
            val editor = sharedPreferences.edit()
            editor.putString("serverIp", ipEditText.text.toString())
            editor.putBoolean("darkTheme", themeSwitch.isChecked)
            editor.apply()

            setResult(Activity.RESULT_OK)

            finish()  // Закрываем экран настроек
        }
    }
}
