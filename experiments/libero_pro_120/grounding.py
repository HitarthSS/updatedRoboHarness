"""Candidate-only SAM3 box recovery using existing image features and VLM box."""
import numpy as np

def _numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

def grounded_box_mask(predictor,state,bbox,image_shape,score_threshold,trace):
    height,width=image_shape
    empty=np.zeros((height,width),dtype=np.uint8)
    trace.update(attempted=False,accepted=False)
    box=np.asarray(bbox,dtype=float)
    if box.shape!=(4,) or not np.isfinite(box).all():return empty,0.0
    box=np.clip(box,[0,0,0,0],[width,height,width,height])
    x1,y1,x2,y2=box
    if x2-x1<2 or y2-y1<2:return empty,0.0
    normalized=[(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height]
    # Reuse image features; remove the failed text prompt so the documented
    # positive-box path supplies its own "visual" prompt.
    predictor.reset_all_prompts(state)
    trace['attempted']=True
    result=predictor.add_geometric_prompt(box=normalized,label=True,state=state)
    scores=_numpy(result.get('scores',[])).reshape(-1)
    masks=_numpy(result.get('masks',[]))
    best=None;best_iou=0.1;best_score=0.0
    for mask,score in zip(masks,scores):
        mask=np.asarray(mask).squeeze()>0
        if mask.shape!=(height,width) or not np.isfinite(score) or score<score_threshold:continue
        yy,xx=np.where(mask)
        if not len(xx):continue
        mx1,my1,mx2,my2=xx.min(),yy.min(),xx.max()+1,yy.max()+1
        intersection=max(0,min(x2,mx2)-max(x1,mx1))*max(0,min(y2,my2)-max(y1,my1))
        union=(x2-x1)*(y2-y1)+(mx2-mx1)*(my2-my1)-intersection
        iou=intersection/union if union>0 else 0.0
        if iou>best_iou:
            best,best_iou,best_score=mask.astype(np.uint8),float(iou),float(score)
    trace.update(accepted=best is not None,box_iou=best_iou if best is not None else None,
                 score=best_score)
    return (best,best_score) if best is not None else (empty,0.0)
