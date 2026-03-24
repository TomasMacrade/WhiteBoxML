"""
Script de validación de docstrings para WhiteBoxML.

Este script utiliza el módulo 'ast' para parsear archivos Python y verificar
que tanto el módulo como sus funciones/clases contengan las etiquetas
obligatorias de autoría y fecha.

:authors: Tomás Macrade
:date: 24/03/2026
"""

import ast
import sys

def check_docstring(docstring, label, path, line_no):
    """
    Valida si las etiquetas existen en un docstring dado.
    
    :param docstring: contenido del string de documentación
    :param label: nombre del elemento (módulo/función) para el error
    :param path: ruta del archivo
    :param line_no: número de línea donde comienza el elemento
    :return: lista de mensajes de error encontrados
    :authors: Tomás Macrade
    :date: 24/03/2026
    """
    errors = []
    if not docstring:
        return [f"{path}:{line_no}: Falta el docstring en {label}."]
    
    if ":authors:" not in docstring:
        errors.append(f"{path}:{line_no}: Falta la etiqueta ':authors:' en {label}")
    if ":date:" not in docstring:
        errors.append(f"{path}:{line_no}: Falta la etiqueta ':date:' en {label}")
    return errors

def check_file(path):
    """
    Parsea un archivo y busca docstrings en módulo, clases y funciones.
    
    :param path: ruta del archivo .py a revisar
    :authors: Tomás Macrade
    :date: 24/03/2026
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return [f"{path}:0: Error de sintaxis al parsear el archivo."]
    
    file_errors = []

    module_doc = ast.get_docstring(tree)
    file_errors.extend(check_docstring(module_doc, "el Módulo", path, 1))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            node_doc = ast.get_docstring(node)
            file_errors.extend(check_docstring(node_doc, f"'{node.name}'", path, node.lineno))
                
    return file_errors

if __name__ == "__main__":
    all_errors = []
    for arg in sys.argv[1:]:
        all_errors.extend(check_file(arg))
    
    if all_errors:
        for err in all_errors:
            print(err)
        sys.exit(1)