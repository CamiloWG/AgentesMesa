import enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import math
import random


class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2


class MyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE
        self.infection_time = None
        self.recovery_time = None

    def move(self):
        if self.model.lockdown_active:
            # si lockdown activo y agente cumple con restricción, no se mueve
            if self.model.lockdown_free_move_prob < 1.0:
                if self.random.random() > self.model.lockdown_free_move_prob:
                    return
        if self.model.movement_reduction > 0:
            # movement_reduction es probabilidad de NO mover
            if self.random.random() < self.model.movement_reduction:
                return

        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def status(self):
        if self.state == State.INFECTED:
            if self.model.death_rate > 0:
                if self.random.random() < self.model.death_rate:
                    # lo retiramos del scheduler y lo marcamos removed
                    self.state = State.REMOVED
                    try:
                        self.model.schedule.remove(self)
                    except Exception:
                        pass
                    return
            # recuperación
            if self.infection_time is not None:
                t = self.model.schedule.time - self.infection_time
                if t >= self.recovery_time:
                    self.state = State.REMOVED

    def contact(self):
        # infecta compañeros de la celda actual
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for other in cellmates:
                if other is self:
                    continue
                if self.state == State.INFECTED and other.state == State.SUSCEPTIBLE:
                    ptrans_eff = self.model.ptrans * self.model.ptrans_multiplier
                    if self.random.random() < ptrans_eff:
                        other.state = State.INFECTED
                        other.infection_time = self.model.schedule.time
                        other.recovery_time = self.model.get_recovery_time()

    def step(self):
        self.status()
        if self.state == State.INFECTED and self.model.test_and_isolate_prob > 0:
            if self.random.random() < self.model.test_and_isolate_prob:
                self.state = State.REMOVED
                return
        self.move()
        self.contact()


class InfectionModel(Model):
    def __init__(
        self,
        N=200,
        width=20,
        height=20,
        ptrans=0.25,
        death_rate=0.0,
        recovery_days=14,
        recovery_sd=3,
        initial_infected_frac=0.02,
        vaccination_frac=0.0,
    ):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.ptrans = ptrans
        self.ptrans_multiplier = 1.0  # factor por intervenciones (mascarillas)
        self.death_rate = death_rate
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.running = True

        self.movement_reduction = 0.0
        self.lockdown_active = False
        self.lockdown_free_move_prob = 0.0
        self.test_and_isolate_prob = 0.0
        self.vaccination_frac = vaccination_frac

        for i in range(self.num_agents):
            a = MyAgent(i, self)
            if random.random() < vaccination_frac:
                a.state = State.REMOVED
            else:
                if random.random() < initial_infected_frac:
                    a.state = State.INFECTED
                    a.infection_time = 0
                    a.recovery_time = self.get_recovery_time()
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(agent_reporters={"State": "state"})

    def get_recovery_time(self):
        val = int(
            max(1, self.random.normalvariate(self.recovery_days, self.recovery_sd))
        )
        return val

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


def get_column_data(model):
    df = model.datacollector.get_agent_vars_dataframe()
    if df.empty:
        return pd.DataFrame(columns=["Step", "Susceptible", "Infected", "Removed"])
    X = pd.pivot_table(
        df.reset_index(), index="Step", columns="State", aggfunc=np.size, fill_value=0
    )

    cols = []
    for c in X.columns:
        if isinstance(c, tuple):
            cols.append(c[-1])
        else:
            cols.append(c)
    X.columns = cols

    rename_map = {
        0: "Susceptible",
        1: "Infected",
        2: "Removed",
        State.SUSCEPTIBLE: "Susceptible",
        State.INFECTED: "Infected",
        State.REMOVED: "Removed",
        "SUSCEPTIBLE": "Susceptible",
        "INFECTED": "Infected",
        "REMOVED": "Removed",
    }
    X = X.rename(columns=rename_map)
    for col in ["Susceptible", "Infected", "Removed"]:
        if col not in X.columns:
            X[col] = 0
    return X[["Susceptible", "Infected", "Removed"]]


def run_single(pop=400, steps=100, **model_kwargs):
    model = InfectionModel(N=pop, width=20, height=20, **model_kwargs)
    for i in range(steps):
        model.step()
    return model


def run_scenario_replicates(scenario_fn, reps=10, **runner_kwargs):
    all_runs = []
    for r in range(reps):
        model = scenario_fn(seed=r, **runner_kwargs)
        steps = runner_kwargs.get("steps", 100)
        for i in range(steps):
            model.step()
        X = get_column_data(model)
        if X.empty:
            X = pd.DataFrame({"Susceptible": [], "Infected": [], "Removed": []})
        X["rep"] = r
        X = X.reset_index()
        all_runs.append(X)
    df_all = pd.concat(all_runs, ignore_index=True)
    return df_all


def scenario_base(seed=0, pop=400, steps=100, **kwargs):
    # escenario base: sin intervenciones
    np.random.seed(seed)
    random.seed(seed)
    model = InfectionModel(
        N=pop,
        width=20,
        height=20,
        ptrans=kwargs.get("ptrans", 0.25),
        death_rate=kwargs.get("death_rate", 0.0),
        recovery_days=kwargs.get("recovery_days", 14),
        recovery_sd=kwargs.get("recovery_sd", 3),
        initial_infected_frac=kwargs.get("initial_infected_frac", 0.02),
        vaccination_frac=kwargs.get("vaccination_frac", 0.0),
    )
    return model


def scenario_masks(seed=0, pop=400, steps=100, mask_effectiveness=0.5, **kwargs):
    model = scenario_base(seed=seed, pop=pop, steps=steps, **kwargs)
    model.ptrans_multiplier = 1.0 - mask_effectiveness
    return model


def scenario_distancing(seed=0, pop=400, steps=100, movement_reduction=0.5, **kwargs):
    model = scenario_base(seed=seed, pop=pop, steps=steps, **kwargs)
    model.movement_reduction = movement_reduction
    return model


def scenario_lockdown(
    seed=0,
    pop=400,
    steps=100,
    lockdown_start=10,
    lockdown_end=40,
    lockdown_free_move_prob=0.05,
    **kwargs
):
    np.random.seed(seed)
    random.seed(seed)
    model = InfectionModel(
        N=pop,
        width=20,
        height=20,
        ptrans=kwargs.get("ptrans", 0.25),
        death_rate=kwargs.get("death_rate", 0.0),
        recovery_days=kwargs.get("recovery_days", 14),
        recovery_sd=kwargs.get("recovery_sd", 3),
        initial_infected_frac=kwargs.get("initial_infected_frac", 0.02),
        vaccination_frac=kwargs.get("vaccination_frac", 0.0),
    )
    model.lockdown_schedule = (lockdown_start, lockdown_end, lockdown_free_move_prob)
    original_step = model.step

    def step_with_lockdown():
        t = model.schedule.time
        start, end, freep = model.lockdown_schedule
        if start <= t < end:
            model.lockdown_active = True
            model.lockdown_free_move_prob = freep
        else:
            model.lockdown_active = False
            model.lockdown_free_move_prob = 1.0
        original_step()

    model.step = step_with_lockdown
    return model


def scenario_vaccination(seed=0, pop=400, steps=100, vaccination_frac=0.2, **kwargs):
    model = InfectionModel(
        N=pop,
        width=20,
        height=20,
        ptrans=kwargs.get("ptrans", 0.25),
        death_rate=kwargs.get("death_rate", 0.0),
        recovery_days=kwargs.get("recovery_days", 14),
        recovery_sd=kwargs.get("recovery_sd", 3),
        initial_infected_frac=kwargs.get("initial_infected_frac", 0.02),
        vaccination_frac=vaccination_frac,
    )
    return model


def scenario_test_and_isolate(seed=0, pop=400, steps=100, test_prob=0.05, **kwargs):
    model = scenario_base(seed=seed, pop=pop, steps=steps, **kwargs)
    model.test_and_isolate_prob = test_prob
    return model


def scenario_combined(seed=0, pop=400, steps=100, **kwargs):
    # Escenario que combina vacunación, mascarillas, distanciamiento y pruebas
    model = InfectionModel(
        N=pop,
        width=20,
        height=20,
        ptrans=kwargs.get("ptrans", 0.25),
        recovery_days=kwargs.get("recovery_days", 14),
        recovery_sd=kwargs.get("recovery_sd", 3),
        initial_infected_frac=kwargs.get("initial_infected_frac", 0.02),
        vaccination_frac=0.30,  # 30% vacunados iniciales
    )
    model.ptrans_multiplier = 0.5  # mascarillas: 50% menos transmisión
    model.movement_reduction = 0.4  # distanciamiento: 40% menos movimiento
    model.test_and_isolate_prob = 0.05  # pruebas y aislamiento diario
    return model


# -----------------------
# Gráficas
# -----------------------
def plot_mean_ci(df_all, title="", ax=None):
    if df_all.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return ax

    grouped = df_all.groupby(["Step"]).agg(
        {
            "Infected": ["mean", "std"],
            "Susceptible": "mean",
            "Removed": "mean",
        }
    )
    grouped.columns = [
        "Infected_mean",
        "Infected_std",
        "Susceptible_mean",
        "Removed_mean",
    ]
    grouped = grouped.reset_index()
    nreps = df_all["rep"].nunique() if "rep" in df_all.columns else 1
    grouped["Infected_ci"] = 1.96 * grouped["Infected_std"] / math.sqrt(max(1, nreps))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(grouped["Step"], grouped["Infected_mean"], label="Infected (mean)")
    ax.fill_between(
        grouped["Step"],
        grouped["Infected_mean"] - grouped["Infected_ci"],
        grouped["Infected_mean"] + grouped["Infected_ci"],
        alpha=0.3,
    )
    ax.plot(grouped["Step"], grouped["Susceptible_mean"], label="Susceptible (mean)")
    ax.plot(grouped["Step"], grouped["Removed_mean"], label="Removed (mean)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return ax


# -----------------------
# Ejemplo de ejecución comparando escenarios
# -----------------------
def example_compare():
    steps = 100
    pop = 400
    reps = 12
    base = run_scenario_replicates(
        lambda **k: scenario_base(**k), reps=reps, pop=pop, steps=steps, ptrans=0.25
    )
    masks = run_scenario_replicates(
        lambda **k: scenario_masks(**k),
        reps=reps,
        pop=pop,
        steps=steps,
        ptrans=0.25,
        mask_effectiveness=0.5,
    )
    distancing = run_scenario_replicates(
        lambda **k: scenario_distancing(**k),
        reps=reps,
        pop=pop,
        steps=steps,
        movement_reduction=0.5,
        ptrans=0.25,
    )
    vaccination = run_scenario_replicates(
        lambda **k: scenario_vaccination(**k),
        reps=reps,
        pop=pop,
        steps=steps,
        vaccination_frac=0.25,
        ptrans=0.25,
    )
    testiso = run_scenario_replicates(
        lambda **k: scenario_test_and_isolate(**k),
        reps=reps,
        pop=pop,
        steps=steps,
        test_prob=0.05,
        ptrans=0.25,
    )

    combined = run_scenario_replicates(
        lambda **k: scenario_combined(**k), reps=12, pop=400, steps=100, ptrans=0.25
    )

    plot_mean_ci(
        combined,
        title="Estrategia combinada (Vacunación + Mascarillas + Distanciamiento + Test)",
    )
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_mean_ci(base, title="Base (sin intervenciones)", ax=axes[0, 0])
    plot_mean_ci(masks, title="Mascarillas (50% efectividad)", ax=axes[0, 1])
    plot_mean_ci(
        distancing, title="Distanciamiento (50% menos movimiento)", ax=axes[1, 0]
    )
    plot_mean_ci(vaccination, title="Vacunaci\u00f3n 25% inicial", ax=axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_mean_ci(testiso, title="Pruebas y Aislamiento (p=0.05 diario)", ax=ax2)
    fig2.tight_layout()
    plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    example_compare()
