# Mapping from Gmsh physical group names to ChargeTransport.jl integer
# boundary region (bregion) indices.
#
# The shared mesh generator (app/prismo/waveguide_mesh.py)
# creates named physical groups: ``silicon``, ``oxide``, ``contact_anode``,
# ``contact_cathode``. ExtendableGrids.simplexgrid("file.msh") discards the
# physical group *names*, so contacts must be wired by integer bregion index.
# ExtendableGrids discards the physical group tags too: it assigns bregion
# indices consecutively in `gmsh.model.getPhysicalGroups(1)` order.  Keep this
# helper in lockstep with that enumeration; a Gmsh tag is not a bregion index
# when, as in the unified mesh spike, tags have gaps.
#
# Ref: ticket 07 (mesh contact mapping layer).

using Gmsh

"""
    get_breking_contacts(mesh_path::String) -> Dict{Symbol,Int}

Read a Gmsh ``.msh`` file and return a ``Dict`` mapping ``:anode`` and
``:cathode`` to the ExtendableGrids boundary-region index of the
``contact_anode`` and ``contact_cathode`` physical groups respectively.

The returned indices can be passed directly to ``set_contact!`` and used to
set ``data.boundaryType[breg] = OhmicContact``.

Returns an empty ``Dict`` when the file does not exist or when Gmsh fails to
read it.
"""
function get_breking_contacts(mesh_path::String)
    contacts = Dict{Symbol,Int}()
    isfile(mesh_path) || return contacts
    try
        gmsh.initialize()
        gmsh.open(mesh_path)
        # Match ExtendableGridsGmshExt.mod_to_simplexgrid: physical-group
        # names become dense ids in `getPhysicalGroups(dim)` order.
        dimTags = gmsh.model.getPhysicalGroups(1)
        for (bregion, (_, tag)) in enumerate(dimTags)
            name = gmsh.model.getPhysicalName(1, tag)
            if name == "contact_anode"
                contacts[:anode] = bregion
            elseif name == "contact_cathode"
                contacts[:cathode] = bregion
            end
        end
        gmsh.clear()
        gmsh.finalize()
    catch
        # Graceful degradation: leave contacts empty on gmsh failure.
        try
            gmsh.finalize()
        catch
        end
    end
    return contacts
end
