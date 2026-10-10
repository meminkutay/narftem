function Model = nartfem_json_to_Model(jsonfile)
% Builds the MatFEA "Model" struct from a nartfem model JSON, with the same conventions as
% ff_run_MatFEA.m (K0 = 0.5, residual stress 3 psi, floor 1000 psi / k4, seed modulus k1*Pa,
% unit weight pcf -> pci, cohesion in psi, sheet stiffness in kN/m).
%
%   Model = nartfem_json_to_Model('examples/FL_AC3_B5_SG4k_GG.json');
%
% The same JSON is read by nartfem.axi in Python, so both engines use identical inputs.

txt     = fileread(jsonfile);
J       = jsondecode(txt);
layers  = J.layers;
if isstruct(layers); layers = num2cell(layers); end       % cell array of layer structs
NL      = numel(layers);

Pa      = 14.706;
if isfield(J, 'solver') && isfield(J.solver, 'atm_pressure_psi'); Pa = J.solver.atm_pressure_psi; end

Model = struct();
Model.General.AtmPressure       = Pa;
Model.Load.TirePressure         = J.load.pressure_psi;
Model.Load.ContactRadius        = J.load.radius_in;
Model.Load.ContactLoad          = Model.Load.TirePressure * pi * Model.Load.ContactRadius ^ 2;

TH = zeros(1, NL-1);
for i = 1:NL-1; TH(i) = layers{i}.thickness_in; end
Model.geometry.LayerHeight      = TH;
Model.geometry.LayerHeightCum   = cumsum(TH);
W = []; D = [];
if isfield(J, 'domain')
    if isfield(J.domain, 'width_in'); W = J.domain.width_in; end
    if isfield(J.domain, 'depth_in'); D = J.domain.depth_in; end
end
if isempty(W); W = 40 * Model.Load.ContactRadius; end
if isempty(D); D = 50 * Model.geometry.LayerHeightCum(end); end
Model.geometry.Width            = W;
Model.geometry.Depth            = D;

% ---- layers
Model.Layers.LayerType          = cell(1, NL);
Model.Layers.MaterialType       = cell(1, NL);
Model.Layers.ResilientModulus   = zeros(1, NL);
Model.Layers.PoissonRatio       = zeros(1, NL);
Model.Layers.UnitWeight         = zeros(1, NL);
Model.Layers.MrPredictiveFunc   = cell(1, NL);
Model.Layers.Kmat               = NaN(4, NL);
Model.Layers.K0                 = 0.5 * ones(1, NL);
Model.Layers.MinMr              = NaN(1, NL);
Model.Layers.ResidualStress     = zeros(1, NL);
Model.Layers.MrRatio            = 0.99 * ones(1, NL);
Model.Layers.PoissonRatioHor    = zeros(1, NL);
Model.Geogrid = struct('Layers', [], 'Location', [], 'InfluenceDepth', [], 'Strn0', [], 'Rho', [], ...
    'Beta', [], 'mm', [], 'nn', [], 'FricitonAngle', [], 'Cohesion', [], 'SheetStiff', [], 'PR', [], 'EquNumCycle', []);

for i = 1:NL
    L = layers{i};
    Model.Layers.LayerType{i}   = L.name;
    Model.Layers.PoissonRatio(i)= L.nu;
    Model.Layers.PoissonRatioHor(i) = L.nu;
    if isfield(L, 'nu_hor') && ~isempty(L.nu_hor); Model.Layers.PoissonRatioHor(i) = L.nu_hor; end
    gw = L.unit_weight_pcf; if gw > 1; gw = gw / 12^3; end
    Model.Layers.UnitWeight(i)  = gw;
    if isfield(L, 'K0') && ~isempty(L.K0); Model.Layers.K0(i) = L.K0; end
    if isfield(L, 'Er_over_Ez') && ~isempty(L.Er_over_Ez); Model.Layers.MrRatio(i) = L.Er_over_Ez; end
    hasGG = isfield(L, 'geogrids') && ~isempty(L.geogrids);
    switch L.type
        case 'elastic'
            Model.Layers.MaterialType{i}    = 'Elastic';
            Model.Layers.ResilientModulus(i)= L.E_psi;
        case 'elastic_depth'
            Model.Layers.MaterialType{i}    = 'Elastic:MrFunc';
            tab = L.E_table;
            % MatFEA evaluates this at the FE coordinate z <= 0 (see Zinew in f_struc_respNonlinGG_MatFEA_V2.m);
            % the table is in positive depth, hence -z.
            Model.Layers.MrPredictiveFunc{i}= @(z) interp1(tab(:,1), tab(:,2), -z, 'linear', 'extrap');
            Model.Layers.ResilientModulus(i)= tab(1,2);
        case {'universal', 'ktheta'}
            if strcmp(L.type, 'universal'); mt = 'Nonlin:Universal'; else; mt = 'Nonlin:KTheta'; end
            if hasGG; mt = [mt, '-GG']; end
            Model.Layers.MaterialType{i}    = mt;
            k = L.k(:);
            Model.Layers.Kmat(1:numel(k), i) = k;
            if isfield(L, 'Mr_min_psi') && ~isempty(L.Mr_min_psi)
                Model.Layers.MinMr(i) = L.Mr_min_psi;
            elseif numel(k) >= 4 && isfinite(k(4))
                Model.Layers.MinMr(i) = k(4);
            else
                Model.Layers.MinMr(i) = 1000;
            end
            Model.Layers.ResidualStress(i)  = 3.0;
            if isfield(L, 'residual_psi') && ~isempty(L.residual_psi); Model.Layers.ResidualStress(i) = L.residual_psi; end
            E0 = 0; if isfield(L, 'E_psi'); E0 = L.E_psi; end
            if ~(E0 > 0)
                if strcmp(L.type, 'ktheta'); E0 = k(1) * Pa ^ k(2); else; E0 = k(1) * Pa; end
            end
            Model.Layers.ResilientModulus(i) = max(E0, Model.Layers.MinMr(i));
        otherwise
            error('Unknown layer type %s', L.type);
    end
    if ~strncmp(Model.Layers.MaterialType{i}, 'Nonlin', 6) && isfield(L, 'residual_psi') && ~isempty(L.residual_psi)
        Model.Layers.ResidualStress(i) = L.residual_psi;
    end
    % ---- geogrids of this layer
    if hasGG
        G = L.geogrids; if isstruct(G); G = num2cell(G); end
        for g = 1:numel(G)
            gg = G{g};
            Model.Geogrid.Layers        = [Model.Geogrid.Layers, i];
            Model.Geogrid.Location      = [Model.Geogrid.Location, gg.l_in];
            Model.Geogrid.InfluenceDepth= [Model.Geogrid.InfluenceDepth, gg.delta_in];
            Model.Geogrid.Strn0         = [Model.Geogrid.Strn0, gg.eps0];
            Model.Geogrid.Rho           = [Model.Geogrid.Rho, gg.rho];
            Model.Geogrid.Beta          = [Model.Geogrid.Beta, gg.beta];
            Model.Geogrid.mm            = [Model.Geogrid.mm, gg.m];
            Model.Geogrid.nn            = [Model.Geogrid.nn, gg.n];
            Model.Geogrid.FricitonAngle = [Model.Geogrid.FricitonAngle, gg.phi_deg];
            Model.Geogrid.Cohesion      = [Model.Geogrid.Cohesion, gg.c_psi];      % psi
            Model.Geogrid.SheetStiff    = [Model.Geogrid.SheetStiff, gg.M_kNm];    % kN/m
            Model.Geogrid.PR            = [Model.Geogrid.PR, gg.vg];
            Model.Geogrid.EquNumCycle   = [Model.Geogrid.EquNumCycle, gg.N];
        end
    end
end

% ---- boundary conditions, mesh, solver, convergence (ff_run_MatFEA defaults)
Model.BC.InnerEdge              = struct('U1', 'fix', 'U2', 'free');
Model.BC.BottomEdge             = struct('U1', 'fix', 'U2', 'fix');
Model.BC.OuterEdge              = struct('U1', 'fix', 'U2', 'free');
Model.Mesh.ImportMesh           = {false, ''};
Model.Mesh.MeshSizeScale        = 1.3;
if isfield(J, 'mesh') && isfield(J.mesh, 'scale') && ~isempty(J.mesh.scale); Model.Mesh.MeshSizeScale = J.mesh.scale; end
Model.Mesh.Mesh4MEAPA           = 0;
Model                           = SideFunc03MeshProperties(Model);
Model.Solver.GaussianPointNumber= 16;
Model.Converge.MaxIteration     = 20;
Model.Converge.Reliability      = 95;
Model.Converge.MaxError         = 2;
if isfield(J, 'solver')
    s = J.solver;
    if isfield(s, 'gauss_points');    Model.Solver.GaussianPointNumber = s.gauss_points; end
    if isfield(s, 'max_iter');        Model.Converge.MaxIteration = s.max_iter; end
    if isfield(s, 'reliability_pct'); Model.Converge.Reliability = s.reliability_pct; end
    if isfield(s, 'max_error_pct');   Model.Converge.MaxError = s.max_error_pct; end
end
Model.Superpose.Enable          = false;
Model.Superpose.DualTireSpace   = 12.0;
Model.Superpose.TandomAxleSpace = 51.6;
Model.Superpose.TridemAxleSpace = 49.2;
Model.Superpose.QuadAxleSpace   = 49.2;
Model.Superpose.PlotEnable      = false;
Model.Plot.Type                 = {};
Model.Plot.MaxRadialCoord       = 5.0 * Model.Load.ContactRadius;
Model.Plot.MaxDepthCoord        = 1.2 * Model.geometry.LayerHeightCum(end);
Model.Plot.ColorMap             = 'jet';
end
