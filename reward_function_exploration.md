# Functions
$E$ is an array of the district buildings' electricity demand, $n$ is the number of buildings in the district and $t$ is the current timestep.

1. __Default__
    $$
    r(t)^\textrm{def} = E(t)
    $$

2. __MARLISA__
    $$
    r(t)^\textrm{marl} = \textrm{sign}\left(r(t)^\textrm{def}\right) 
                            \times 0.01 
                                \times \left|{r(t)^\textrm{def}}\right|^2 
                                        \times \textrm{max}\left(
                                            0, 
                                            -\sum_{i=1}^{n-1}{r(t)^\textrm{def}_i}
                                        \right)

    $$

3. __SAC__
    $$
    r(t)^\textrm{sac} = \left[
        \textrm{min}\left(
            0,
            {r(t)_0^\textrm{def}}^3
        \right),
        \dots,
        \textrm{min}\left(
            0,
            {r(t)_{n-1}^\textrm{def}}^3
        \right)
    \right]
    $$

4. __Ramping Square__
    $$
    r(t)^\textrm{ramp} = \left[
        -\left(\left(
            \sum_{i=0}^{n-1}{r(t)^\textrm{def}_i} - \sum_{i=0}^{n-1}{r(t-1)^\textrm{def}_i}
        \right)^2\right)_0,
        \dots,
        -\left(\left(
            \sum_{i=0}^{n-1}{r(t)^\textrm{def}_i} - \sum_{i=0}^{n-1}{r(t-1)^\textrm{def}_i}
        \right)^2\right)_{n-1},
    \right] \div n
    $$

5. __Exponential__
    $$
    r(t)^\textrm{exp} = \left[
        \left(-e^{0.0002 \times \sum_{i=0}^{n-1}{-r(t)^\textrm{def}_i}}\right)_0,
        \dots,
        \left(-e^{0.0002 \times \sum_{i=0}^{n-1}{-r(t)^\textrm{def}_i}}\right)_{n-1}
    \right] \div n
    $$

    Where 0.0002 is a scaling factor to avoid high exponents defined as $\frac{\lambda}{50}$. We use $\lambda = 0.01$.

6. __Mixed__
    $$
    r(t)^\textrm{mix} = r(t)^\textrm{ramp} + r(t)^\textrm{exp}
    $$