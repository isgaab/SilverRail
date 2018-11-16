from app import create_app
import sys

#args_num = len(sys.argv)

#if sys.argv[0] == 'uwsgi':
app = create_app()

if __name__ == '__main__':
    app.run()
