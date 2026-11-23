"""Convert a CAD STEP file to a mesh Gazebo can draw: binary glTF (.glb, keeps the CAD colours) and STL (no colours).
Units are converted to metres (STEP lengths are read in mm by OpenCascade). Prints the bounding box, so the model's
axes can be lined up with the body frame (x forward, y left, z up, origin at the centre of mass).

    uv run --with cadquery-ocp --with trimesh python models/gz/step_to_mesh.py IN.step OUT_STEM [--defl 0.3]
"""
import argparse

from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.Message import Message_ProgressRange
from OCP.RWGltf import RWGltf_CafWriter, RWGltf_WriterTrsfFormat
from OCP.RWMesh import RWMesh_CoordinateSystem
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.OCP.collections import IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString as InfoMap
from OCP.TDocStd import TDocStd_Document
from OCP.OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("out")
    ap.add_argument("--defl", type=float, default=0.3, help="linear deflection in mm (smaller = finer mesh)")
    a = ap.parse_args()
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    rd = STEPCAFControl_Reader()
    rd.SetColorMode(True); rd.SetNameMode(True); rd.SetLayerMode(True)
    if rd.ReadFile(a.step) != IFSelect_RetDone:
        raise SystemExit(f"cannot read {a.step}")
    rd.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    st.GetFreeShapes(labels)
    comp = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(comp)
    for i in range(1, labels.Length() + 1):
        bb.Add(comp, st.GetShape_s(labels.Value(i)))
    BRepMesh_IncrementalMesh(comp, a.defl, False, 0.3, True)
    StlAPI_Writer().Write(comp, a.out + "_mm.stl")
    w = RWGltf_CafWriter(TCollection_AsciiString(a.out + ".glb"), True)
    cs = RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup
    w.ChangeCoordinateSystemConverter().SetInputLengthUnit(0.001)       # mm in
    w.ChangeCoordinateSystemConverter().SetInputCoordinateSystem(cs)
    w.ChangeCoordinateSystemConverter().SetOutputLengthUnit(1.0)        # m out
    w.ChangeCoordinateSystemConverter().SetOutputCoordinateSystem(cs)
    w.SetTransformationFormat(RWGltf_WriterTrsfFormat.RWGltf_WriterTrsfFormat_Compact)
    w.Perform(doc, InfoMap(), Message_ProgressRange())
    m = trimesh.load(a.out + "_mm.stl")
    m.apply_scale(0.001)
    m.export(a.out + ".stl")
    print(f"{len(labels)} top shapes, {len(m.faces)} triangles; bounds (m): min {m.bounds[0].round(4).tolist()} "
          f"max {m.bounds[1].round(4).tolist()} size {m.extents.round(4).tolist()}")


if __name__ == "__main__":
    main()
