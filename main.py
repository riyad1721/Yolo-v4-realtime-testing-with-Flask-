from website import create_app

app = create_app()

if __name__ == '__main__':
    #app.run(debug=True)
    app.debug = True
    app.run(host='172.168.23.105', port=8081)
