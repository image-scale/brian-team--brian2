"""
Synaptic connections between neuron groups.

Provides the Synapses class for modeling synaptic connections with
pre-synaptic spike effects, weights, and delays.
"""

import numpy as np

from neurosim.core.base import SimObject
from neurosim.core.variables import Variables, ArrayVariable, DynamicArrayVariable, Constant
from neurosim.equations import Equations, PARAMETER, DIFFERENTIAL_EQUATION
from neurosim.units import second
from neurosim.units.core import DIMENSIONLESS, Quantity, get_dimensions
from neurosim.utils.logger import get_logger

__all__ = ['Synapses']

logger = get_logger(__name__)


class SynapsePathway(SimObject):
    """
    Handles spike propagation for a set of synapses.

    Executes code (e.g., on_pre) when source neurons spike.
    """

    def __init__(self, synapses, prepost, code, delay=None):
        name = f'{synapses.name}_{prepost}'
        super().__init__(name=name, clock=synapses.clock)

        self.synapses = synapses
        self.prepost = prepost
        self.code = code
        self.when = 'synapses'

        if delay is not None:
            if isinstance(delay, Quantity):
                delay = float(np.asarray(delay))
            self._delay = delay
        else:
            self._delay = 0.0

        self._queue = {}

    def run(self):
        """Process spikes and execute pathway code."""
        t = self.clock.t_
        dt = self.clock.dt_

        if self.prepost == 'pre':
            source = self.synapses.source
        else:
            source = self.synapses.target

        spikes = source.spikes

        if len(spikes) > 0:
            if self._delay > 0:
                delivery_time = t + self._delay
                delivery_step = int(round(delivery_time / dt))
                if delivery_step not in self._queue:
                    self._queue[delivery_step] = []
                self._queue[delivery_step].extend(spikes.tolist())
            else:
                self._process_spikes(spikes)

        current_step = int(round(t / dt))
        if current_step in self._queue:
            delayed_spikes = np.array(self._queue.pop(current_step), dtype=np.int32)
            if len(delayed_spikes) > 0:
                self._process_spikes(delayed_spikes)

    def _process_spikes(self, source_spikes):
        """Execute pathway code for given source spikes."""
        if self.code is None:
            return

        syn_idx = self.synapses._get_synapses_for_source(
            source_spikes, self.prepost
        )

        if len(syn_idx) == 0:
            return

        namespace = self.synapses._get_namespace(syn_idx)

        for stmt in self.code.split('\n'):
            stmt = stmt.strip()
            if not stmt or stmt.startswith('#'):
                continue

            if '+=' in stmt:
                var_name, expr = stmt.split('+=', 1)
                var_name = var_name.strip()
                expr = expr.strip()

                value = eval(expr, {"__builtins__": {}}, namespace)
                if isinstance(value, Quantity):
                    value = np.asarray(value)

                target_var = var_name
                if var_name.endswith('_post'):
                    target_var = var_name[:-5]
                elif var_name.endswith('_pre'):
                    target_var = var_name[:-4]

                if target_var in self.synapses.target.variables:
                    post_indices = self.synapses.variables['j'].get_value()[syn_idx]
                    target_data = self.synapses.target.variables[target_var].get_value()
                    np.add.at(target_data, post_indices, value)
                elif var_name in self.synapses.variables:
                    syn_data = self.synapses.variables[var_name].get_value()
                    syn_data[syn_idx] += value

            elif '=' in stmt:
                var_name, expr = stmt.split('=', 1)
                var_name = var_name.strip()
                expr = expr.strip()

                value = eval(expr, {"__builtins__": {}}, namespace)
                if isinstance(value, Quantity):
                    value = np.asarray(value)

                target_var = var_name
                if var_name.endswith('_post'):
                    target_var = var_name[:-5]
                elif var_name.endswith('_pre'):
                    target_var = var_name[:-4]

                if target_var in self.synapses.target.variables:
                    post_indices = self.synapses.variables['j'].get_value()[syn_idx]
                    target_data = self.synapses.target.variables[target_var].get_value()
                    target_data[post_indices] = value
                elif var_name in self.synapses.variables:
                    syn_data = self.synapses.variables[var_name].get_value()
                    syn_data[syn_idx] = value


class Synapses(SimObject):
    """
    Synaptic connections between neuron groups.

    Parameters
    ----------
    source : NeuronGroup
        The pre-synaptic neuron group.
    target : NeuronGroup, optional
        The post-synaptic neuron group. Defaults to source (self-connections).
    model : str or Equations, optional
        Equations defining synaptic variables.
    on_pre : str, optional
        Code to execute when pre-synaptic neuron spikes.
    on_post : str, optional
        Code to execute when post-synaptic neuron spikes.
    delay : Quantity, optional
        Synaptic delay.
    namespace : dict, optional
        Additional namespace for equations.
    name : str, optional
        Name of this synapse group.

    Examples
    --------
    >>> from neurosim.groups import NeuronGroup
    >>> from neurosim.synapses import Synapses
    >>> pre = NeuronGroup(10, 'dv/dt = 1*mV/ms : volt', threshold='v > 10*mV', reset='v = 0*mV')
    >>> post = NeuronGroup(10, 'dv/dt = 0 : volt')
    >>> S = Synapses(pre, post, on_pre='v_post += 1*mV')
    >>> S.connect(j='i')  # One-to-one connection
    """

    def __init__(self, source, target=None, model=None, on_pre=None, on_post=None,
                 delay=None, namespace=None, name='synapses*'):
        super().__init__(name=name, clock=source.clock)

        self.source = source
        self.target = target if target is not None else source
        self._namespace = namespace or {}
        self.when = 'synapses'

        self.variables = Variables(owner=self)

        self.variables.add_dynamic_array('i', dtype=np.int32)
        self.variables.add_dynamic_array('j', dtype=np.int32)

        if model is not None:
            self.equations = Equations(model) if isinstance(model, str) else model
            self._create_variables_from_equations()
        else:
            self.equations = Equations('')

        self._pathways = []
        if on_pre is not None:
            pathway = SynapsePathway(self, 'pre', on_pre, delay)
            self._pathways.append(pathway)

        if on_post is not None:
            pathway = SynapsePathway(self, 'post', on_post)
            self._pathways.append(pathway)

        self._contained_objects = list(self._pathways)

        self._N = 0

        logger.debug(f"Created Synapses '{self.name}' from {source.name} to {self.target.name}")

    def _create_variables_from_equations(self):
        """Create synaptic variables from equations."""
        for name, eq in self.equations.items():
            if name in self.variables:
                continue

            if eq.type == PARAMETER:
                self.variables.add_dynamic_array(name, dimensions=eq.dim)
            elif eq.type == DIFFERENTIAL_EQUATION:
                self.variables.add_dynamic_array(name, dimensions=eq.dim)

    @property
    def N(self):
        """Number of synapses."""
        return self._N

    def __len__(self):
        return self._N

    def connect(self, condition=None, i=None, j=None, p=None, n=None):
        """
        Create synaptic connections.

        Parameters
        ----------
        condition : str, optional
            Boolean condition for connections (uses i, j indices).
        i : str or array, optional
            Pre-synaptic indices or expression.
        j : str or array, optional
            Post-synaptic indices or expression.
        p : float, optional
            Connection probability.
        n : int, optional
            Number of synapses per connection.

        Examples
        --------
        Connect all to all:
        >>> S.connect()

        One-to-one:
        >>> S.connect(j='i')

        Probabilistic:
        >>> S.connect(p=0.1)

        Specific connections:
        >>> S.connect(i=[0, 1, 2], j=[3, 4, 5])
        """
        n_pre = self.source.N
        n_post = self.target.N

        if condition is not None:
            raise NotImplementedError("Condition-based connect not yet implemented")

        if j is not None and isinstance(j, str):
            all_i = []
            all_j = []
            for pre_idx in range(n_pre):
                ns = {'i': pre_idx, 'N_pre': n_pre, 'N_post': n_post}
                j_val = eval(j, {"__builtins__": {}}, ns)
                if isinstance(j_val, (int, np.integer)):
                    if 0 <= j_val < n_post:
                        all_i.append(pre_idx)
                        all_j.append(int(j_val))
                else:
                    for jj in j_val:
                        if 0 <= jj < n_post:
                            all_i.append(pre_idx)
                            all_j.append(int(jj))

            pre_indices = np.array(all_i, dtype=np.int32)
            post_indices = np.array(all_j, dtype=np.int32)

        elif i is not None and j is not None:
            pre_indices = np.asarray(i, dtype=np.int32)
            post_indices = np.asarray(j, dtype=np.int32)

        elif p is not None:
            conn = np.random.rand(n_pre, n_post) < p
            pre_indices, post_indices = np.nonzero(conn)
            pre_indices = pre_indices.astype(np.int32)
            post_indices = post_indices.astype(np.int32)

        else:
            pre_indices, post_indices = np.meshgrid(
                np.arange(n_pre), np.arange(n_post), indexing='ij'
            )
            pre_indices = pre_indices.ravel().astype(np.int32)
            post_indices = post_indices.ravel().astype(np.int32)

        if n is not None and n > 1:
            pre_indices = np.repeat(pre_indices, n)
            post_indices = np.repeat(post_indices, n)

        self._add_synapses(pre_indices, post_indices)

    def _add_synapses(self, pre_indices, post_indices):
        """Add synapses with given pre and post indices."""
        n_new = len(pre_indices)
        n_old = self._N

        self.variables['i'].set_value(np.concatenate([
            self.variables['i'].get_value(), pre_indices
        ]))
        self.variables['j'].set_value(np.concatenate([
            self.variables['j'].get_value(), post_indices
        ]))

        for name, var in self.variables.items():
            if name in ('i', 'j'):
                continue
            if isinstance(var, DynamicArrayVariable):
                var.resize(n_old + n_new)

        self._N = n_old + n_new

        logger.debug(f"Added {n_new} synapses to '{self.name}', total: {self._N}")

    def _get_synapses_for_source(self, source_indices, prepost):
        """Get synapse indices where source matches given indices."""
        if prepost == 'pre':
            source_var = 'i'
        else:
            source_var = 'j'

        all_sources = self.variables[source_var].get_value()
        mask = np.isin(all_sources, source_indices)
        return np.nonzero(mask)[0]

    def _get_namespace(self, syn_idx):
        """Build namespace for evaluating synapse code."""
        ns = dict(self._namespace) if self._namespace else {}

        ns['t'] = self.clock.t_
        ns['dt'] = self.clock.dt_

        for name, var in self.variables.items():
            if isinstance(var, Constant):
                ns[name] = var.get_value()
            elif isinstance(var, (ArrayVariable, DynamicArrayVariable)):
                data = var.get_value()[syn_idx]
                if var.dim != DIMENSIONLESS:
                    data = Quantity(data, dim=var.dim)
                ns[name] = data

        for name, var in self.target.variables.items():
            if isinstance(var, ArrayVariable):
                post_indices = self.variables['j'].get_value()[syn_idx]
                data = var.get_value()[post_indices]
                if var.dim != DIMENSIONLESS:
                    data = Quantity(data, dim=var.dim)
                ns[f'{name}_post'] = data
                ns[name] = data

        for name, var in self.source.variables.items():
            if isinstance(var, ArrayVariable):
                pre_indices = self.variables['i'].get_value()[syn_idx]
                data = var.get_value()[pre_indices]
                if var.dim != DIMENSIONLESS:
                    data = Quantity(data, dim=var.dim)
                ns[f'{name}_pre'] = data

        ns['exp'] = np.exp
        ns['log'] = np.log
        ns['sin'] = np.sin
        ns['cos'] = np.cos
        ns['sqrt'] = np.sqrt
        ns['abs'] = np.abs
        ns['clip'] = np.clip

        from neurosim import units
        for attr in dir(units):
            obj = getattr(units, attr)
            if isinstance(obj, (Quantity, float, int)) and not attr.startswith('_'):
                ns[attr] = obj

        return ns

    @property
    def i(self):
        """Pre-synaptic neuron indices."""
        return self.variables['i'].get_value()

    @property
    def j(self):
        """Post-synaptic neuron indices."""
        return self.variables['j'].get_value()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        if name in ('source', 'target', 'variables', 'equations', 'clock',
                    'name', 'id', 'when', 'order', 'active'):
            raise AttributeError(name)

        use_units = not name.endswith('_')
        var_name = name[:-1] if name.endswith('_') else name

        if not hasattr(self, 'variables') or var_name not in self.variables:
            raise AttributeError(f"'{var_name}' is not a synaptic variable")

        var = self.variables[var_name]
        values = var.get_value()

        if use_units and var.dim != DIMENSIONLESS:
            return Quantity(values, dim=var.dim)
        return values

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('source', 'target', 'variables',
                                             'equations', 'clock', 'name', 'id',
                                             'when', 'order', 'active'):
            object.__setattr__(self, name, value)
            return

        if hasattr(self, 'variables') and name in self.variables:
            var = self.variables[name]

            if isinstance(value, Quantity):
                value = np.asarray(value)

            if np.isscalar(value):
                value = np.full(self._N, value)

            var.set_value(value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return f"<Synapses '{self.name}' from {self.source.name} to {self.target.name}, {self._N} synapses>"
