# First define custom types for dispatch
# @userplot KalmanFilterTruthPlot
# @userplot KalmanSmootherTruthPlot
# @userplot KalmanFilterPlot
# @userplot KalmanSmootherPlot


"""
    @recipe function f(::Type{Val{:myribbon}}, x, lower, upper)
Define a custom series recipe called :myribbon which plots
a ribbon between `lower` and `upper` values at each point `x`.
"""
@recipe function f(::Type{Val{:myribbon}}, x, lower, upper)
    # We tell Plots.jl we want to use its built-in ribbon logic
    seriestype := :ribbon
    
    # Optionally, set some defaults for the fill color, line color, etc.
    fillcolor := :pink
    linecolor := :red
    
    # Return the triple of vectors that define the ribbon:
    # (x-coordinates, lower boundary, upper boundary)
    (x, lower, upper)
end

"""
    KalmanFilterTruthPlot{T,M<:FilterOutput{<:Real}}


A type for plotting Kalman filter results against true states.

# Fields
- `true_states::T`: The true state values to compare against
- `results::M`: The output from running the Kalman filter (`FilterOutput`)

This type is used internally by the plotting recipes to create comparison plots
between the true states and the Kalman filter estimates, including confidence intervals.
"""
struct KalmanFilterTruthPlot{T,M <: FilterOutput{<:Real}}
    true_states::T
    results::M
end

"""
    KalmanSmootherTruthPlot{T,M<:SmootherOutput{<:Real}}

A type for plotting Kalman smoother results against true states.

# Fields
- `true_states::T`: The true state values to compare against
- `results::M`: The output from running the Kalman smoother (`SmootherOutput`)

This type is used internally by the plotting recipes to create comparison plots
between the true states and the Kalman smoother estimates, including confidence intervals.
"""
struct KalmanSmootherTruthPlot{T,M <: SmootherOutput{<:Real}}
    true_states::T
    results::M
end

"""
    KalmanFilterPlot{M<:FilterOutput{<:Real}}

A type for plotting Kalman filter results.

# Fields
- `results::M`: The output from running the Kalman filter (`FilterOutput`)

This type is used internally by the plotting recipes to create visualizations
of the Kalman filter estimates and their confidence intervals.
"""
struct KalmanFilterPlot{M <: FilterOutput{<:Real}}
    results::M
end

"""
    KalmanSmootherPlot{M<:SmootherOutput{<:Real}}

A type for plotting Kalman smoother results.

# Fields
- `results::M`: The output from running the Kalman smoother (`SmootherOutput`)

This type is used internally by the plotting recipes to create visualizations
of the Kalman smoother estimates and their confidence intervals.
"""
struct KalmanSmootherPlot{M <: SmootherOutput{<:Real}}
    results::M
end

# Define plot recipes
"""
    @recipe f(kf::KalmanFilterTruthPlot)

Recipe for plotting Kalman filter results against true states.

Creates a multi-panel plot comparing the true states with their Kalman filter estimates.
Each state dimension gets its own subplot showing:
- True state trajectory (black solid line)
- Filtered state estimate (tomato dashed line) 
- 95% confidence intervals (tomato shaded region)

# Arguments
- `kf::KalmanFilterTruthPlot`: Container with true states and filter results

# Plot Attributes
- `title`: "Kalman Filter Performance" 
- `layout`: One subplot per state dimension
- `size`: 1000×(350*N) pixels where N is number of states
- `legend`: Positioned at bottom
- `gridalpha`: 0.3 for subtle grid lines

The plot uses consistent styling:
- True states: Black solid lines
- Filter estimates: Tomato dashed lines
- Confidence intervals: Light tomato shaded regions
"""
@recipe function f(kf::KalmanFilterTruthPlot)
    X, results = kf.true_states, kf.results
    N, T = size(X)
    X_filter = results.Z_tt[1:N, :]
    P_filter = results.P_tt[1:N, 1:N, :]

    # Set default plot attributes
    title --> "Kalman Filter Performance"
    linewidth --> 2
    label --> ["True State $i" for i in 1:N]
    color --> :lightcoral
    layout := (N, 1)
    size --> (1000, 350*N)
    legend := :bottom
    legendcolumn := 3
    gridalpha --> 0.1  # Lighter grid lines for a softer look
    
    # Create series for each state
    for i in 1:N
        ci = 1.96 .* sqrt.(reshape(P_filter[i,i,:], :))
        
        @series begin
            subplot := i
            seriestype := :path
            linecolor --> :lightcoral
            linestyle --> :dash
            label := "Kalman Filter $i"
            primary := true
            1:T, X_filter[i,:]
        end
        
        @series begin
            subplot := i
            seriestype := :path
            ribbon := ci          # Tells Plots to shade ±ci around the center
            fillalpha --> 0.15    # Softer fill for the confidence interval
            linealpha --> 0.8     # Slightly transparent line for a softer look
            label := "95% CI"
            color --> :lightcoral
            1:T, X_filter[i,:]
        end
        
        @series begin
            subplot := i
            seriestype := :path
            linecolor --> :black
            linestyle --> :dash
            label := "True State $i"
            primary := true
            1:T, X[i,:]
        end
    end
end

"""
    @recipe f(ks::KalmanSmootherTruthPlot)

Recipe for plotting Kalman smoother results against true states.

Creates a multi-panel plot comparing the smoothed state estimates to the true states.
Each state dimension gets its own subplot showing:
- True state trajectory in black
- Smoothed state estimate in blue dashed line 
- 95% confidence intervals as blue shaded regions

# Arguments
- `ks::KalmanSmootherTruthPlot`: Container with true states and smoother results

# Returns
A plot recipe that will create a vertically stacked set of subplots, one for each state dimension.

The plot includes:
- Title "Kalman Smoother Performance"
- Legend at bottom
- Consistent styling with blue colors for estimates
- 95% confidence intervals based on smoothed state covariances
"""
@recipe function f(ks::KalmanSmootherTruthPlot)
    X, results = ks.true_states, ks.results
    N, T = size(X)
    X_smooth = results.Z_smooth[1:N, :]
    P_smooth = results.P_smooth[1:N, 1:N, :]

    # Set default plot attributes
    title --> "Kalman Smoother Performance"
    linewidth --> 2
    color --> :dodgerblue
    layout := (N, 1)
    size --> (1000, 350*N)
    legend := :bottom
    legendcolumn := 3
    gridalpha --> 0.3
    
    # Create series for each state
    for i in 1:N
        ci = 1.96 .* sqrt.(reshape(P_smooth[i,i,:], :))
        
        @series begin
            subplot := i
            seriestype := :path
            linecolor --> :dodgerblue
            linestyle --> :dash
            label := "Kalman Smoothed $i"
            primary := true
            1:T, X_smooth[i,:]
        end

        
        @series begin
            subplot := i
            seriestype := :path
            ribbon := ci          # Tells Plots to shade ±ci around the center
            fillalpha --> 0.2
            linealpha --> 1.0
            label := "95% CI"
            color --> :tomato
            primary := true
            1:T, X_smooth[i,:]
        end
        

        @series begin
            subplot := i
            seriestype := :path
            linecolor --> :black
            linestyle --> :dash
            label := "True State $i"
            primary := true
            1:T, X[i,:]
        end
    end
end

"""
    @recipe f(kf::KalmanFilterPlot)

Recipe for plotting Kalman filter results.

Creates a multi-panel plot showing the filtered state estimates.
Each state dimension gets its own subplot showing:
- Filtered state estimate (tomato dashed line) 
- 95% confidence intervals (tomato shaded region)

# Arguments
- `kf::KalmanFilterPlot`: Container with filter results

# Plot Attributes
- `title`: "Kalman Filter Estimates" 
- `layout`: One subplot per state dimension
- `size`: 1000×(350*N) pixels where N is number of states
- `legend`: Positioned at bottom
- `gridalpha`: 0.3 for subtle grid lines

The plot uses consistent styling:
- Filter estimates: Tomato dashed lines
- Confidence intervals: Light tomato shaded regions
"""
@recipe function f(kf::KalmanFilterPlot)
    results = kf.results
    N = size(results.Z_tt, 1)
    T = size(results.Z_tt, 2)
    X_filter = results.Z_tt[1:N, :]
    P_filter = results.P_tt[1:N, 1:N, :]

    # Set default plot attributes
    title --> "Kalman Filter Estimates"
    linewidth --> 2
    layout := (N, 1)
    size --> (1000, 350*N)
    legend := :bottom
    legendcolumn := 2
    gridalpha --> 0.3
    

    # Create series for each state
    for i in 1:N
        ci = 1.96 .* sqrt.(reshape(P_filter[i,i,:], :))
        
        @series begin
            subplot := i
            seriestype := :path
            linecolor --> :tomato
            linestyle --> :dash
            label := "State $i Estimate"
            1:T, X_filter[i,:]
        end
        
        @series begin
            subplot := i
            seriestype := :path
            ribbon := ci          # Tells Plots to shade ±ci around the center
            fillalpha --> 0.2
            linealpha --> 1.0
            label := "95% CI"
            color --> :tomato
            1:T, X_filter[i,:]
        end
    end
end

"""
    @recipe f(ks::KalmanSmootherPlot)

Recipe for plotting Kalman smoother results.

Creates a multi-panel plot showing the smoothed state estimates.
Each state dimension gets its own subplot showing:
- Smoothed state estimate in blue dashed line 
- 95% confidence intervals as blue shaded regions

# Arguments
- `ks::KalmanSmootherPlot`: Container with smoother results

# Returns
A plot recipe that will create a vertically stacked set of subplots, one for each state dimension.

The plot includes:
- Title "Kalman Smoother Estimates"
- Legend at bottom
- Consistent styling with blue colors for estimates
- 95% confidence intervals based on smoothed state covariances
"""
@recipe function f(ks::KalmanSmootherPlot)
    results = ks.results
    N = size(results.Z_smooth, 1)
    T = size(results.Z_smooth, 2)
    X_smooth = results.Z_smooth[1:N, :]
    P_smooth = results.P_smooth[1:N, 1:N, :]

    # Set default plot attributes
    title --> "Kalman Smoother Estimates"
    linewidth --> 2
    layout := (N, 1)
    size --> (1000, 350*N)
    legend := :bottom
    gridalpha --> 0.3
    
    # Create series for each state
    for i in 1:N
        ci = 1.96 .* sqrt.(reshape(P_smooth[i,i,:], :))
        
        @series begin
            subplot := i
            seriestype := :path
            linecolor --> :dodgerblue
            label := "State $i Estimate"
            1:T, X_smooth[i,:]
        end
        
        @series begin
            subplot := i
            seriestype := :path
            ribbon := ci          # Tells Plots to shade ±ci around the center
            fillalpha --> 0.2
            linealpha --> 1.0
            label := "95% CI"
            color --> :dodgerblue
            1:T, X_smooth[i,:]
        end
    end
end


"""
    kalman_filter_truth_plot(X, results)

Create a plot comparing true states with Kalman filter estimates.

# Arguments
- `X`: Matrix of true state values (N×T)
- `results`: FilterOutput object containing filtered state estimates

# Returns
A plot showing true states, filtered estimates, and confidence intervals
"""
kalman_filter_truth_plot(X, results) = KalmanFilterTruthPlot(X, results)

"""
    kalman_smoother_truth_plot(X, results)

Create a plot comparing true states with Kalman smoother estimates.

# Arguments
- `X`: Matrix of true state values (N×T)
- `results`: SmootherOutput object containing smoothed state estimates


# Returns
A plot showing true states, smoothed estimates, and confidence intervals
"""
kalman_smoother_truth_plot(X, results) = KalmanSmootherTruthPlot(X, results)

"""
    kalman_filter_plot(results)

Create a plot of Kalman filter estimates.

# Arguments
- `results`: FilterOutput object containing filtered state estimates

# Returns
A plot showing filtered estimates and confidence intervals
"""
kalman_filter_plot(results) = KalmanFilterPlot(results)

"""
    kalman_smoother_plot(results)

Create a plot of Kalman smoother estimates.

# Arguments
- `results`: SmootherOutput object containing smoothed state estimates

# Returns
A plot showing smoothed estimates and confidence intervals
"""
kalman_smoother_plot(results) = KalmanSmootherPlot(results)