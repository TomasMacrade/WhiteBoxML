# Fundamentos Tecnicos

Este documento describe los componentes matematicos y estructurales incorporados en la implementacion de modelos lineales dentro del proyecto WhiteBoxML.

## BaseLinearModel

`BaseLinearModel` es una clase abstracta que define la interfaz comun para todos los modelos lineales del proyecto.

Su objetivo es unificar el manejo de parametros (`weights` y `bias`) y establecer los metodos que cualquier modelo lineal debe implementar, como:

- `fit`
- `predict`
- `compute_gradient`
- `compute_hessian`

Ademas, esta clase incorpora metodos genericos de optimizacion reutilizables:

- Gradient Descent
- Newton Method

Gracias a esta estructura, distintos modelos pueden reutilizar la misma logica de optimizacion sin duplicar codigo.

---

## Gradient Descent

Gradient Descent (Descenso por Gradiente) es un algoritmo de optimizacion de primer orden utilizado para minimizar una funcion de costo de manera iterativa.

En cada iteracion, el algoritmo calcula el gradiente de la funcion de error y actualiza los parametros en direccion opuesta al gradiente.

La actualizacion general puede expresarse como:

w = w - lr * dw

donde:

- `w` representa los parametros del modelo
- `lr` es el learning rate
- `dw` es el gradiente

Este metodo es ampliamente utilizado debido a su simplicidad y eficiencia computacional en problemas con grandes volumenes de datos.

Sin embargo, requiere seleccionar adecuadamente el learning rate para garantizar convergencia estable.

---

## Newton Method

Newton Method (Metodo de Newton) es un algoritmo de optimizacion de segundo orden.

A diferencia de Gradient Descent, utiliza no solo el gradiente sino tambien la informacion de curvatura de la funcion mediante el Hessiano (segunda derivada).

La actualizacion general se expresa como:

w = w - H^-1 * gradiente

donde:

- `H` es la matriz Hessiana
- `gradiente` representa la primera derivada de la funcion de costo

Este enfoque suele converger mas rapidamente que Gradient Descent y, en problemas cuadraticos como regresion lineal, puede alcanzar el minimo en muy pocas iteraciones.

Su principal desventaja es el costo computacional asociado al calculo e inversion de la matriz Hessiana.

---

## Diferencias entre Gradient Descent y Newton Method

La principal diferencia entre ambos algoritmos radica en la informacion matematica utilizada para encontrar el minimo de la funcion de costo.

Gradient Descent:

- Utiliza solamente el gradiente
- Es un metodo de primer orden
- Requiere learning rate
- Es computacionalmente eficiente
- Escala bien en grandes volumenes de datos

Newton Method:

- Utiliza gradiente y Hessiano
- Es un metodo de segundo orden
- No requiere learning rate
- Converge mas rapidamente
- Tiene mayor costo computacional

En general, Gradient Descent resulta mas conveniente para problemas de gran escala, mientras que Newton Method ofrece mayor presicion y velocidad de convergencia en problemas de tamaño moderado.

---

## Autor

- Cristian

## Fecha

2026-05-06
