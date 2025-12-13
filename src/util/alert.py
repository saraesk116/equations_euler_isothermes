import sys


def incomplete(func_name):
    print("Erreur : vous devez compléter le corps de la fonction '" + func_name + "'")
    sys.exit()


def error(msg):
    print("Erreur : " + msg)
    sys.exit()
