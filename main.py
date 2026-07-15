import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

print("Librerías cargadas correctamente")


promedio = ctrl.Antecedent(np.arange(0, 21, 1), "promedio")

asistencia = ctrl.Antecedent(np.arange(0, 101, 1), "asistencia")

participacion = ctrl.Antecedent(np.arange(0, 11, 1), "participacion")


prioridad_beca = ctrl.Consequent(np.arange(0, 101, 1), "prioridad_beca")


promedio["bajo"] = fuzz.trapmf(promedio.universe, [0, 0, 10, 13])
promedio["medio"] = fuzz.trimf(promedio.universe, [11, 14, 17])
promedio["alto"] = fuzz.trapmf(promedio.universe, [15, 18, 20, 20])

asistencia["baja"] = fuzz.trapmf(asistencia.universe, [0, 0, 50, 65])
asistencia["media"] = fuzz.trimf(asistencia.universe, [55, 75, 90])
asistencia["alta"] = fuzz.trapmf(asistencia.universe, [80, 90, 100, 100])


participacion["baja"] = fuzz.trapmf(participacion.universe, [0, 0, 3, 5])
participacion["media"] = fuzz.trimf(participacion.universe, [4, 6, 8])
participacion["alta"] = fuzz.trapmf(participacion.universe, [7, 9, 10, 10])


prioridad_beca["baja"] = fuzz.trapmf(
    prioridad_beca.universe,
    [0, 0, 30, 45]
)

prioridad_beca["media"] = fuzz.trimf(
    prioridad_beca.universe,
    [35, 55, 75]
)

prioridad_beca["alta"] = fuzz.trapmf(
    prioridad_beca.universe,
    [65, 80, 100, 100]
)

def plot_membership(variable, titulo):

    plt.figure(figsize=(8,4))

    for nombre, termino in variable.terms.items():

        plt.plot(variable.universe,
                 termino.mf,
                 label=nombre)

    plt.title(titulo)
    plt.xlabel("Universo")
    plt.ylabel("Grado de pertenencia")
    plt.grid(True)
    plt.legend()

    plt.show()


plot_membership(promedio,"Promedio")
plot_membership(asistencia,"Asistencia")
plot_membership(participacion,"Participación")
plot_membership(prioridad_beca,"Prioridad de beca")



student_promedio = 16
student_asistencia = 85
student_participacion = 7

values = {
    "promedio_bajo": fuzz.interp_membership(
        promedio.universe,
        promedio["bajo"].mf,
        student_promedio,
    ),
    "promedio_medio": fuzz.interp_membership(
        promedio.universe,
        promedio["medio"].mf,
        student_promedio,
    ),
    "promedio_alto": fuzz.interp_membership(
        promedio.universe,
        promedio["alto"].mf,
        student_promedio,
    ),
    "asistencia_baja": fuzz.interp_membership(
        asistencia.universe,
        asistencia["baja"].mf,
        student_asistencia,
    ),
    "asistencia_media": fuzz.interp_membership(
        asistencia.universe,
        asistencia["media"].mf,
        student_asistencia,
    ),
    "asistencia_alta": fuzz.interp_membership(
        asistencia.universe,
        asistencia["alta"].mf,
        student_asistencia,
    ),
    "participacion_baja": fuzz.interp_membership(
        participacion.universe,
        participacion["baja"].mf,
        student_participacion,
    ),
    "participacion_media": fuzz.interp_membership(
        participacion.universe,
        participacion["media"].mf,
        student_participacion,
    ),
    "participacion_alta": fuzz.interp_membership(
        participacion.universe,
        participacion["alta"].mf,
        student_participacion,
    ),
}

for nombre, valor in values.items():
    print(f"{nombre}: {valor:.2f}")


rule_1 = ctrl.Rule(
    promedio["alto"] & asistencia["alta"],
    prioridad_beca["alta"]
)

rule_2 = ctrl.Rule(
    promedio["alto"] & participacion["alta"],
    prioridad_beca["alta"]
)

rule_3 = ctrl.Rule(
    promedio["medio"] & asistencia["alta"],
    prioridad_beca["media"]
)

rule_4 = ctrl.Rule(
    promedio["medio"] & participacion["media"],
    prioridad_beca["media"]
)

rule_5 = ctrl.Rule(
    promedio["bajo"] & asistencia["baja"],
    prioridad_beca["baja"]
)

rule_6 = ctrl.Rule(
    promedio["bajo"] & participacion["baja"],
    prioridad_beca["baja"]
)

rule_7 = ctrl.Rule(
    asistencia["media"] & participacion["alta"],
    prioridad_beca["media"]
)

rule_8 = ctrl.Rule(
    promedio["alto"] &
    asistencia["media"] &
    participacion["media"],
    prioridad_beca["alta"]
)

rule_9 = ctrl.Rule(
    promedio["medio"] &
    asistencia["media"] &
    participacion["baja"],
    prioridad_beca["media"]
)

rule_10 = ctrl.Rule(
    promedio["bajo"] &
    asistencia["alta"] &
    participacion["alta"],
    prioridad_beca["media"]
)

rules = [
    rule_1,
    rule_2,
    rule_3,
    rule_4,
    rule_5,
    rule_6,
    rule_7,
    rule_8,
    rule_9,
    rule_10,
]

print(f"\nSe definieron {len(rules)} reglas.")



scholarship_control = ctrl.ControlSystem(rules)

scholarship_simulator = ctrl.ControlSystemSimulation(
    scholarship_control
)

print("Sistema difuso creado correctamente.")




scholarship_simulator.input["promedio"] = 16
scholarship_simulator.input["asistencia"] = 85
scholarship_simulator.input["participacion"] = 7

scholarship_simulator.compute()

resultado = scholarship_simulator.output["prioridad_beca"]

print("\nResultado final")
print(f"Prioridad de beca: {resultado:.2f}")


prioridad_beca.view(sim=scholarship_simulator)

plt.title("Resultado defuzzificado")

plt.show()



def evaluate_student(promedio_value,
                     asistencia_value,
                     participacion_value):

    simulator = ctrl.ControlSystemSimulation(
        scholarship_control
    )

    simulator.input["promedio"] = promedio_value
    simulator.input["asistencia"] = asistencia_value
    simulator.input["participacion"] = participacion_value

    simulator.compute()

    score = simulator.output["prioridad_beca"]

    if score < 40:
        categoria = "Baja prioridad"

    elif score < 70:
        categoria = "Prioridad media"

    else:
        categoria = "Alta prioridad"

    return {
        "promedio": promedio_value,
        "asistencia": asistencia_value,
        "participacion": participacion_value,
        "prioridad_score": round(score, 2),
        "categoria": categoria,
    }
print("\nPrueba de la función:")
print(evaluate_student(16, 85, 7))




students = [
    (19, 95, 9),
    (16, 85, 7),
    (13, 78, 6),
    (11, 60, 4),
    (9, 45, 2),
    (12, 95, 9),
    (18, 70, 5)
]

results = [evaluate_student(*student) for student in students]

results_df = pd.DataFrame(results)

print("\nResultados de los estudiantes:")
print(results_df)



plt.figure(figsize=(9,5))

plt.bar(
    results_df.index.astype(str),
    results_df["prioridad_score"]
)

plt.title("Prioridad de beca por estudiante")
plt.xlabel("Estudiante")
plt.ylabel("Prioridad")
plt.ylim(0,100)
plt.grid(axis="y")

plt.show()


def rigid_rule(promedio_value,
               asistencia_value,
               participacion_value):

    if (
        promedio_value >= 16 and
        asistencia_value >= 80 and
        participacion_value >= 7
    ):
        return "Alta prioridad"

    return "No alta prioridad"


results_df["regla_rigida"] = results_df.apply(
    lambda row: rigid_rule(
        row["promedio"],
        row["asistencia"],
        row["participacion"]
    ),
    axis=1
)

print("\nComparación con la regla rígida:")
print(results_df)



sensitivity_results = []

for promedio_actual in range(0,21):

    evaluacion = evaluate_student(
        promedio_actual,
        85,
        7
    )

    sensitivity_results.append(evaluacion)

sensitivity_df = pd.DataFrame(sensitivity_results)
plt.figure(figsize=(9,5))

plt.plot(
    sensitivity_df["promedio"],
    sensitivity_df["prioridad_score"],
    marker="o"
)

plt.title("Análisis de sensibilidad")
plt.xlabel("Promedio académico")
plt.ylabel("Prioridad de beca")

plt.ylim(0,100)

plt.grid(True)

plt.show()


print("\nPrimeros resultados del análisis de sensibilidad:")

print(sensitivity_df.head())

print("\n===================================")
print("LABORATORIO COMPLETADO EXITOSAMENTE")
print("===================================")