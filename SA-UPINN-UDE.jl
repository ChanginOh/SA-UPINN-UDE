using CSV, DataFrames, Flux, Optim, JLD2
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode
using SciMLSensitivity
using Random, Statistics, Dates

# ==================================================
# SA-UPINN with UDE Refinement (SA-UPINN-UDE)
#
# Author: Changin Oh
# Date: November 25, 2026
# Publication: Manuscript in preparation.
# GitHub: https://github.com/ChanginOh
# ==================================================

# ==================================================
# 1. User settings
# ==================================================
# Set random seed for reproducibility
Random.seed!(9999)

# Set Adam training parameters
adamEpochs = 100000
adamLearningRate = 0.0001

# Set UDE training parameters
useUDE = true
udeIterations = 5000

# Set SA weighting parameters
useSAWeighting = true
saLearningRate = 0.001
saWeightMax = 10.0

# Set loss weights for non-SA weighting
dataLossWeight = 1.0
odeLossWeight = 1000.0
icLossWeight = 10.0

# Set display frequency for training progress
dispIterations = 5000

# Set neural network architecture parameters (with tanh activation functions)
numHiddenLayers = 2
numNeuronsSolutionNN = 8
numNeuronsInteractionNN = 8

numColPoints = 200

# Define mask function to weight ODE and IC losses adaptively
function maskFunction(omega)
    # return 2 / (1 + exp(-2 * (omega - 1)))
    return 1*omega^3
end

# ==================================================
# 2. Dataset
# ==================================================
# Load the dataset and sort by time
dataTable = CSV.read(joinpath(@__DIR__, "Dataset", "UT_mean.csv"), DataFrame)
sort!(dataTable, :Time)

# Scale time inputs to [0, 1] for better training
observationTimes = Float64.(dataTable.Time)
timeStart = first(observationTimes)
timeScale = last(observationTimes) - timeStart
observationInputs = reshape((observationTimes .- timeStart) ./ timeScale, 1, :)

# Create list of observed cell populations for each resistant percentage
resistantPercentages = [0, 10, 25, 50, 75, 90, 100]
observedCells = [Float64.(permutedims(hcat(dataTable[!, "S($(percent)%)"], dataTable[!, "R($(percent)%)"]))) for percent in resistantPercentages]

# Create normalized collocation points
collocationTimes = collect(range(first(observationTimes), last(observationTimes); length=numColPoints))
collocationInputs = reshape((collocationTimes .- timeStart) ./ timeScale, 1, :)

# Define initial adaptive weights for loss function
adaptiveWeights = nothing
if useSAWeighting
    # adaptiveWeights = ones(Float64, length(resistantPercentages), numColPoints + 1)
    adaptiveWeights = rand(Float64, length(resistantPercentages), numColPoints + 1)
end

# ==================================================
# 3. Neural networks
# ==================================================
# Define solution neural networks for each resistant percentage
solutionNetworks = Tuple(Flux.f64(Chain(Dense(1 => numNeuronsSolutionNN, tanh), (Dense(numNeuronsSolutionNN => numNeuronsSolutionNN, tanh) for _ in 2:numHiddenLayers)..., Dense(numNeuronsSolutionNN => 2))) for _ in resistantPercentages)

# Define interaction neural network for interaction terms
interactionNetwork = Flux.f64(Chain(Dense(2 => numNeuronsInteractionNN, tanh), (Dense(numNeuronsInteractionNN => numNeuronsInteractionNN, tanh) for _ in 2:numHiddenLayers)..., Dense(numNeuronsInteractionNN => 2)))

neuralNetworks = (solutionNetworks=solutionNetworks, interactionNetwork=interactionNetwork)

# ==================================================
# 4. ODE model
# ==================================================
# Define ODE parameters for untreated tumour growth
rₛ, rᵣ = 0.03267820555398815, 0.03384136479474916
Kₛ, Kᵣ = 1.106440597333263, 0.7412391064500214

# Define tumour growth model with NN interaction terms
function tumourODE(u, p, t)
    S, R = u
    NN = p
    interactionTerms = S * R .* NN(u)

    return [rₛ * S * (1 - S / Kₛ) + interactionTerms[1], rᵣ * R * (1 - R / Kᵣ) + interactionTerms[2]]
end

# ==================================================
# 5. Loss functions
# ==================================================
function loss(networks, adaptiveWeights=nothing; returnComponents=false)
    totalError = 0.0
    totalDataError, totalRawDataError = 0.0, 0.0
    totalODEError, totalRawODEError = 0.0, 0.0
    totalICError, totalRawICError = 0.0, 0.0

    for datasetIndex in eachindex(observedCells)
        # Extract solution network for current dataset
        solutionNetwork = networks.solutionNetworks[datasetIndex]

        # Compute predicted observations and raw data and IC errors
        predictedObservations = solutionNetwork(observationInputs)
        rawDataError = mean(abs2, predictedObservations .- observedCells[datasetIndex])
        rawICError = mean(abs2, predictedObservations[:, 1] .- observedCells[datasetIndex][:, 1])

        # Compute derivatives of predicted observations with respect to time using chain rule
        hiddenOutput = collocationInputs
        hiddenDerivative = ones(size(collocationInputs))

        for layer in solutionNetwork.layers[1:end-1]
            hiddenOutput = layer(hiddenOutput)
            hiddenDerivative = (1 .- hiddenOutput.^2) .* (layer.weight * hiddenDerivative)
        end

        predictedCells = solutionNetwork.layers[end](hiddenOutput)
        predictedTimeDerivatives = (solutionNetwork.layers[end].weight * hiddenDerivative) ./ timeScale

        # Compute ODE error
        S = predictedCells[1:1, :]
        R = predictedCells[2:2, :]
        interactionTerms = (S .* R) .* networks.interactionNetwork(predictedCells)

        odeDerivativeS = rₛ .* S .* (1 .- S ./ Kₛ) .+ interactionTerms[1:1, :]
        odeDerivativeR = rᵣ .* R .* (1 .- R ./ Kᵣ) .+ interactionTerms[2:2, :]

        odeResiduals = predictedTimeDerivatives .- vcat(odeDerivativeS, odeDerivativeR)
        rawODEError = mean(abs2, odeResiduals)

        # Compute weighted errors based on whether adaptive weights are used
        if adaptiveWeights === nothing
            dataError = dataLossWeight * rawDataError
            odeError = odeLossWeight * rawODEError
            icError = icLossWeight * rawICError
        else
            dataError = rawDataError
            squaredResiduals = vec(sum(abs2, odeResiduals; dims=1))
            mask = maskFunction.(adaptiveWeights[datasetIndex, 1:numColPoints])
            odeError = sum(mask .* squaredResiduals) / (2 * numColPoints)
            icError = maskFunction(adaptiveWeights[datasetIndex, end]) * rawICError
        end

        # Compute total error for current dataset
        totalError += dataError + odeError + icError

        totalDataError += dataError
        totalRawDataError += rawDataError
        totalODEError += odeError
        totalRawODEError += rawODEError
        totalICError += icError
        totalRawICError += rawICError
    end

    # Normalize total error
    normalization = length(resistantPercentages)
    finalLoss = totalError / normalization

    # Return loss information if requested
    if returnComponents
        return (finalLoss = finalLoss, L_Data = totalDataError / normalization, L_Data_raw = totalRawDataError / normalization, L_ODE = totalODEError / normalization, L_ODE_raw = totalRawODEError / normalization, L_IC = totalICError / normalization, L_IC_raw = totalRawICError / normalization)
    end

    return finalLoss
end

# Define UDE loss function for refining interaction network parameters
function udeLoss(parameters, rebuildInteraction, data)
    total = 0.0
    rhs = (u, p, t) -> tumourODE(u, rebuildInteraction(p), t)
    for observations in data.observedCells
        problem = ODEProblem(rhs, observations[:, 1], data.timeSpan, parameters)
        solution = solve(problem, Tsit5(); saveat=data.observationTimes, abstol=1e-8, reltol=1e-6)
        successful_retcode(solution) && length(solution.t) == length(data.observationTimes) || return Inf

        prediction = Array(solution)
        all(isfinite, prediction) || return Inf

        total += mean(abs2, prediction .- observations)
    end
    return total / length(data.observedCells)
end

# ==================================================
# 6. Adam training
# ==================================================
# Set up Adam optimizers for solution NNs and adaptive weights
adamState = Flux.setup(Flux.Adam(adamLearningRate), neuralNetworks)
saAdamState = useSAWeighting ? Flux.setup(Flux.Adam(saLearningRate), adaptiveWeights) : nothing

startTime = time()

# Train solution NNs and adaptive weights using Adam optimizer
for epoch in 1:adamEpochs
    if useSAWeighting
        lossValue, lossGradients = Flux.withgradient(loss, neuralNetworks, adaptiveWeights)
        Flux.update!(adamState, neuralNetworks, lossGradients[1])
        Flux.update!(saAdamState, adaptiveWeights, -lossGradients[2])
        
        # Prevent adaptive weights from overflowing
        # clamp!(adaptiveWeights, 0.0, saWeightMax)
    else
        lossValue, lossGradients = Flux.withgradient(loss, neuralNetworks)
        Flux.update!(adamState, neuralNetworks, lossGradients[1])
    end

    # Display training progress
    if epoch % dispIterations == 0
        info = loss(neuralNetworks, adaptiveWeights; returnComponents=true)
        elapsedTime = round((time() - startTime) / 60; digits=2)

        println("Adam epoch = ", epoch, " | Loss = ", round(info.finalLoss; sigdigits=5), " | Data = ", round(info.L_Data; sigdigits=5), " | ODE = ", round(info.L_ODE; sigdigits=5), " | IC = ", round(info.L_IC; sigdigits=5), " | Elapsed = ", elapsedTime, " mins")

        if useSAWeighting
            println("    Data raw = ", round(info.L_Data_raw; sigdigits=5), " |ODE raw = ", round(info.L_ODE_raw; sigdigits=5), " | IC raw = ", round(info.L_IC_raw; sigdigits=5))
        end
    end
end

# ==================================================
# 7. UDE refinement
# ==================================================
# Refine interaction network parameters using L-BFGS optimizer
if useUDE
    # Obtain initial interaction network parameters and reconstruction function
    initialInteractionWeights, rebuildInteraction = Flux.destructure(neuralNetworks.interactionNetwork)

    # Prepare training data for UDE loss function
    trainingData = (observedCells=observedCells, timeSpan=(first(observationTimes), last(observationTimes)), observationTimes=observationTimes)
    udeObjective = p -> udeLoss(p, rebuildInteraction, trainingData)
    
    # Compute initial UDE loss before refinement
    beforeLoss = udeObjective(initialInteractionWeights)
    isfinite(beforeLoss) || error("Initial UDE loss is not finite: $beforeLoss")
    println("Initial UDE loss = ", beforeLoss)

    # Optimize interaction network parameters
    finalParameters = copy(initialInteractionWeights)

    udeGradient = p -> Flux.gradient(udeObjective, p)[1]

    result = Optim.optimize(udeObjective, udeGradient, copy(initialInteractionWeights), Optim.LBFGS(), Optim.Options(iterations=udeIterations, g_abstol=1e-8, show_trace=true, show_every=dispIterations); inplace=false)

    candidate = Optim.minimizer(result)

    if udeObjective(candidate) <= beforeLoss
        finalParameters = copy(candidate)
    end

    # Compute final UDE loss and rebuild interaction network with refined parameters
    afterLoss = udeObjective(finalParameters)
    interactionNN = rebuildInteraction(finalParameters)

    println("Final UDE loss = ", afterLoss)
else
    interactionNN = neuralNetworks.interactionNetwork
end

# ==================================================
# 8. Training summary
# ==================================================
weightInfo = useSAWeighting ? "SA weighting" : "Fixed loss weights: Data = $(dataLossWeight), ODE = $(odeLossWeight), IC = $(icLossWeight)"

lossInfo = loss(neuralNetworks, adaptiveWeights; returnComponents=true)

trainingInfo =
    "\n=============================\n" *
    "Adam epochs: $(adamEpochs) | Adam learning rate: $(adamLearningRate)\n" *
    "Collocation points: $(numColPoints) | Hidden layers: $(numHiddenLayers) | Neurons per hidden layer: solution NN = $(numNeuronsSolutionNN), interaction NN = $(numNeuronsInteractionNN)\n" *
    "Loss = $(lossInfo.finalLoss) | L_Data = $(lossInfo.L_Data) | L_ODE = $(lossInfo.L_ODE) | L_IC = $(lossInfo.L_IC)\n" *
    "$(weightInfo)\n"

if useSAWeighting
    trainingInfo *= "SA learning rate = $(saLearningRate) | SA weight max = $(saWeightMax)"
    trainingInfo *= "\nL_ODE (raw) = $(lossInfo.L_ODE_raw) | L_IC (raw) = $(lossInfo.L_IC_raw)"
end

if useUDE
    trainingInfo *= "\nUDE loss: $(beforeLoss) → $(afterLoss) | UDE iterations: $(Optim.iterations(result)) / $(udeIterations)"
end

println(trainingInfo)

# ==================================================
# 9. Interaction network parameter saving
# ==================================================
outputDirectory = joinpath(@__DIR__, "Results")
mkpath(outputDirectory)

savedTime = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
parameterFile = joinpath(outputDirectory, "interactionNN_parameters_$(savedTime).jld2")

JLD2.jldsave(parameterFile; weights=[copy(layer.weight) for layer in interactionNN.layers], biases=[copy(layer.bias) for layer in interactionNN.layers])
println("Parameter file saved: ", parameterFile)
