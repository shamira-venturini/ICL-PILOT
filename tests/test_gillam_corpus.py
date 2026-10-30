from pathlib import Path
import hashlib,json
import pandas as pd
import pytest

from icl_pilot.evaluation.story_text_exports import _gillam_child_metadata,_select_gillam_sessions,build_gillam_external_text_records
from icl_pilot.chat.corpus_archives import original_source_bytes

ROOT=Path(__file__).resolve().parents[1]


def chat(age="6;03.",sex="male",group="SLI",stories=("McDonalds","LateForSchool","Aliens")):
    return "@ID:\teng|Gillam|CHI|"+age+"|"+sex+"|||Target_Child|||\n@Types:\tcross, narrative, "+group+"\n" + \
           "".join("@G:\t"+story+"\n*CHI:\ther her goed .\n" for story in stories)


def test_header_age_and_sex_override_folder_bins(tmp_path):
    result=_gillam_child_metadata(chat("5;11.","female"),tmp_path/"SLI/6m/child.cha",tmp_path)
    assert (result["cohort_year"],result["age_months"],result["sex"])==(5,71,"female")


def test_unknown_months_are_not_replaced_with_zero(tmp_path):
    result=_gillam_child_metadata(chat("9;"),tmp_path/"SLI/9m/child.cha",tmp_path)
    assert result["age_months"] is None and result["cohort_year"]==9


def test_conflicting_group_metadata_stops(tmp_path):
    with pytest.raises(ValueError,match="@Types clinical group conflicts"):
        _gillam_child_metadata(chat(group="TD"),tmp_path/"SLI/6m/child.cha",tmp_path)


def test_repeat_visit_is_one_child_and_earliest_visit_is_selected(tmp_path):
    visits=[]
    for session,age,date in [("a-1-68","6;03.","24-JUL-2002"),("a-1-68pre","5;10.","16-MAR-2002")]:
        raw=chat(age,"female")+"@Date:\t"+date+"\n@Comment:\tDOB = 4/18/1996\n"
        visits.append(_gillam_child_metadata(raw,tmp_path/"SLI"/(session+".cha"),tmp_path))
    selected,repeats=_select_gillam_sessions(visits)
    assert len(selected)==1 and selected[0]["source_session_id"]=="a-1-68pre"
    assert selected[0]["source_child_id"]=="a-1-68"
    assert repeats[0]["excluded_session_ids"]==["a-1-68"]
    visits[1]["birth_date_raw"]="1/1/1996"
    with pytest.raises(ValueError,match="Conflicting or unverified"):
        _select_gillam_sessions(visits)


def test_different_child_ids_are_not_merged_on_birth_date_alone():
    visits=[{"source_child_id":child,"group":"SLI","sex":"male","birth_date_raw":"7/2/1996"}
            for child in ("a-2-03","a-2-04")]
    selected,repeats=_select_gillam_sessions(visits)
    assert len(selected)==2 and not repeats


def test_export_retains_partial_SLI_and_ignores_empty_trailing_marker(tmp_path):
    source=tmp_path/"raw"
    sli=source/"SLI/6m/s.cha";td=source/"TD/6m/t.cha"
    sli.parent.mkdir(parents=True);td.parent.mkdir(parents=True)
    sli.write_text(chat(stories=("McDonalds","Aliens")))
    td.write_text(chat(group="TD")+"@G:\tAliens\n@End\n")
    artifacts=build_gillam_external_text_records(project_root=tmp_path,transcripts_dir=source,output_csv="external.csv")
    frame=pd.read_csv(artifacts["output_csv"])
    assert frame.groupby("group").size().to_dict()=={"SLI":2,"TD":3}
    assert frame["transcript_path"].str.startswith("raw/").all()
    summary=json.loads(artifacts["summary_json"].read_text())
    assert len(summary["selection"]["ignored_empty_story_markers"])==1
    with pytest.raises(FileExistsError):
        build_gillam_external_text_records(project_root=tmp_path,transcripts_dir=source,output_csv="external.csv")


def test_original_Gillam_selection_has_182_unique_children_and_543_stories():
    path=ROOT/"data/inputs/gillam/external_story_records.csv"
    if not path.exists():pytest.skip("Gillam source/export are unversioned data inputs")
    frame=pd.read_csv(path,dtype={"source_child_id":str})
    children=frame.drop_duplicates("source_child_id")
    assert len(children)==182 and children["group"].value_counts().to_dict()=={"TD":91,"SLI":91}
    assert len(frame)==543 and frame["group"].value_counts().to_dict()=={"TD":273,"SLI":270}
    assert (children["age_months"].min(),children["age_months"].max())==(60,95)
    strata=children.groupby(["cohort_year","sex","group"]).size().unstack("group")
    assert strata["TD"].equals(strata["SLI"])
    assert not frame.duplicated(["source_child_id","story_id"]).any()
    assert set(frame.loc[frame["source_child_id"].eq("a-1-68"),"source_session_id"])=={"a-1-68pre"}
    partial=frame.groupby(["group","source_child_id"]).size()
    assert (partial==2).sum()==3 and partial.loc["TD"].eq(3).all()
    for row in frame.itertuples(index=False):
        assert hashlib.sha256(original_source_bytes(ROOT/row.transcript_path, ROOT)).hexdigest()==row.transcript_sha256
