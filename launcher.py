import os, sys, time, subprocess

def run_command():
    # Функция запуска команды
    return subprocess.Popen(["cmd", "/c", "lt --port 5000 --subdomain iskra-ai-server"],
                            shell=True,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)

def start_server():
    return subprocess.Popen([f"{os.path.dirname(sys.argv[0])}\\server.exe", "1"],
                            shell=True,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)

print('Starting server...')
serverprocess = start_server()

time.sleep(5)
print("Server started!")

print('Starting listener...')
process = run_command()
print('Listener started!')

while True:
    if process.poll() is not None:  # Если процесс завершился
        print("Listener down. Restarting in 5 seconds...")

        time.sleep(5)

        # Перезапускаем процесс
        process = run_command()
    else:
        # Если процесс всё ещё работает, подождём немного
        time.sleep(1)