function out = run_matfea_dump(jsonfile, dumpfile)
% Runs MatFEA (MATLAB) on a nartfem model JSON and writes a JSON dump that the Python script
% scripts/compare_with_matlab.py reads: the mesh (so Python can run on the identical mesh), the
% first linear solution, the moduli after the first update and the converged results.
%
%   run_matfea_dump('examples/FL_AC3_B5_SG4k_GG.json')                 -> <json>_matfea_dump.json
%   run_matfea_dump('examples/FL_AC3_B5_SG4k_GG.json', 'mydump.json')
%
% Requirements: this file and nartfem_json_to_Model.m on the path together with the MatFEA
% V4 folder (MatFEA.m, Define*.m, SideFunc03MeshProperties.m ...).

if nargin < 2 || isempty(dumpfile)
    [fld, nm] = fileparts(jsonfile);
    dumpfile  = fullfile(fld, [nm, '_matfea_dump.json']);
end
tic
Model   = nartfem_json_to_Model(jsonfile);

% ---- same sequence as MatFEA.m, keeping the Model struct
Model   = DefineGenerateMesh(Model);
Model   = DefineLoadMesh(Model);
mesh.Nodes          = Model.Mesh.Nodes;
mesh.Connectivity   = Model.Mesh.Connectivity;
mesh.EleLyrNum      = Model.Mesh.EleLyrNum(:);
mesh.EleCenter      = Model.Mesh.EleCenter;
Model   = DefineCalculations(Model);
Model   = DefineForceMatrix(Model);
Model   = DefineConstrains(Model);
Model   = DefineSolver(Model);
lin.nodal_disp      = full(Model.Calc.NodalDisp(:));
lin.force_total     = full(sum(Model.Calc.Force));
lin.elem_Mr_seed    = [Model.Calc.Elements.Mr]';
Model   = DefineResponses(Model);
it1.elem_Mr         = [Model.Calc.Elements.Mr]';
it1.elem_strain     = cat(2, Model.Calc.Elements.Strain)';
it1.elem_stress     = cat(2, Model.Calc.Elements.Stress)';
it1.elem_delsig     = [Model.Calc.Elements.delSigHor]';
it1.MrError         = Model.Calc.MrError(:);
Model   = DefineNonlinearity(Model);

fin.elem_Mr         = [Model.Calc.Elements.Mr]';
fin.elem_strain     = cat(2, Model.Calc.Elements.Strain)';
fin.elem_stress     = cat(2, Model.Calc.Elements.Stress)';
fin.elem_delsig     = [Model.Calc.Elements.delSigHor]';
fin.nodal_disp      = full(Model.Calc.NodalDisp(:));
fin.converged       = Model.Calc.Converged;
fin.iterations      = Model.Calc.ConvergeIter;
fin.conv_error      = Model.Calc.ConvergeError;
if isfield(Model.Calc, 'LayerMrSummary'); fin.LayerMrSummary = Model.Calc.LayerMrSummary; end
fin.elem_center     = cat(1, Model.Calc.Elements.Center);
fin.elem_layer      = [Model.Calc.Elements.LayerNum]';

dump.source         = jsonfile;
dump.matlab_version = version;
dump.mesh           = mesh;
dump.linear         = lin;
dump.iteration1     = it1;
dump.final          = fin;
dump.geometry       = Model.geometry;
dump.elapsed_s      = toc;

fid = fopen(dumpfile, 'w'); fwrite(fid, jsonencode(dump), 'char'); fclose(fid);
fprintf('MatFEA dump written: %s  (%d nodes, %d elements, %.1f s)\n', dumpfile, ...
        size(mesh.Nodes, 1), size(mesh.Connectivity, 1), dump.elapsed_s);
out = dump;
end
