select count(distinct P.Id)
from Posts as P, PostTags as PT, Tags as T
where P.PostTypeId=1 and P.Id=PT.PostId and PT.TagId=T.Id and T.TagName='c#'
and P.CreationDate<'2013-01-01'


select count(distinct P1.Id)
from Posts as P1, Posts as P2, PostTags as PT, Tags as T
where P1.PostTypeId=2 and P1.ParentId=P2.Id and P2.Id=PT.PostId and PT.TagId=T.Id and T.TagName='assembly'
and P1.CreationDate<'2013-01-01'


