# Mapping from Gmsh physical group names to ChargeTransport.jl integer
# boundary region (bregion) indices.
#
# The shared mesh generator (app/prismo/waveguide_mesh.py)
# creates named physical groups: ``silicon``, ``oxide``, ``contact_anode``,
# ``contact_cathode``. ExtendableGrids.simplexgrid("file.msh") discards the
# physical group *names*, so contacts must be wired by integer bregion index.
# For boundary entities (dim = dimension-1) the physical group tag equals the
# ExtendableGrids bregion index.
#
# Ref: ticket 07 (mesh contact mapping layer).

using Gmsh

"""
    get_breking_contacts(mesh_path::String) -> Dict{Symbol,Int}

Read a Gmsh ``.msh`` file and return a ``Dict`` mapping ``:anode`` and
``:cathode`` to the integer physical group tag of the ``contact_anode`` and
``contact_cathode`` physical groups respectively.

The physical group tag of a boundary entity equals the ExtendableGrids bregion
index, so the returned values can be passed directly to ``set_contact!`` and
used to set ``data.boundaryType[breg] = OhmicContact``.

Returns an empty ``Dict`` when the file does not exist or when Gmsh fails to
read it.
"""
function get_breking_contacts(mesh_path::String)
    contacts = Dict{Symbol,Int}()
    isfile(mesh_path) || return contacts
    try
        gmsh.initialize()
        gmsh.open(mesh_path)
        dimTags = gmsh.model.getPhysicalGroups()
        i = 1
        while i <= length(dimTags)
            dim = dimTags[i]
            tag = dimTags[i + 1]
            name = gmsh.model.getPhysicalName(dim, tag)
            if name == "contact_anode"
                contacts[:anode] = tag
            elseif name == "contact_cathode"
                contacts[:cathode] = tag
            end
            i += 2
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
